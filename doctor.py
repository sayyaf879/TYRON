"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TYRON DOCTOR  —  External Watchdog & Self-Healing Guardian        ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║      python doctor.py                                                        ║
║                                                                              ║
║  WHAT IT DOES:                                                               ║
║  • Starts TYRON (run.py) as a child subprocess                              ║
║  • Watches TYRON's stdout/stderr in real-time                               ║
║  • If TYRON crashes → Doctor stays alive, analyzes the crash,              ║
║    calls Groq AI to generate a fix, tests it, applies it, restarts TYRON   ║
║  • Doctor NEVER shares memory/threads with TYRON — so TYRON's crash        ║
║    cannot kill Doctor                                                        ║
║  • Doctor only exits when YOU press Ctrl+C                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────┐
  │  doctor.py  (this file — runs as its own process forever)   │
  │                                                             │
  │  ┌───────────────────────────────────────────────────────┐  │
  │  │  TYRON  (python run.py — child subprocess)            │  │
  │  │   ↓ crashes with error                                │  │
  │  └───────────────────────────────────────────────────────┘  │
  │               ↓                                             │
  │   Doctor detects crash (process exit code != 0)             │
  │               ↓                                             │
  │   Doctor sends crash to Groq AI → gets fix code            │
  │               ↓                                             │
  │   Doctor tests fix in isolated subprocess                   │
  │               ↓  (pass)                                     │
  │   Doctor applies fix (backup + replace file)                │
  │               ↓                                             │
  │   Doctor restarts TYRON fresh                               │
  └─────────────────────────────────────────────────────────────┘
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
DOCTOR_DIR      = BASE_DIR / "database" / "doctor"
DOCTOR_LOG_FILE = DOCTOR_DIR / "doctor_log.json"
BACKUP_DIR      = DOCTOR_DIR / "backups"
SANDBOX_DIR     = DOCTOR_DIR / "sandbox"
CRASH_LOG_FILE  = DOCTOR_DIR / "last_crash.txt"

PKT = timezone(timedelta(hours=5))

MAX_RESTART_ATTEMPTS = 20          # Max total restarts before Doctor gives up
RESTART_BACKOFF_BASE = 3           # Seconds between restarts (doubles each time, max 60)
MAX_BACKOFF          = 60          # Max seconds between restarts
SANDBOX_TIMEOUT      = 20          # Seconds for sandbox test
MAX_FIX_ATTEMPTS     = 3           # AI fix attempts per crash
CRASH_BUFFER_LINES   = 200         # How many stderr lines to keep for analysis
MIN_UPTIME_HEALTHY   = 30          # If TYRON runs > 30s, reset restart counter

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP  (Doctor has its own logger, independent of TYRON)
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "[DOCTOR %(asctime)s] %(levelname)s — %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("DOCTOR")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(PKT).isoformat()


def _ensure_dirs():
    DOCTOR_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


def _load_env() -> Dict[str, str]:
    """Read .env file manually (no dotenv dependency needed here)."""
    env_path = BASE_DIR / ".env"
    result: Dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _get_groq_keys(env: Dict[str, str]) -> List[str]:
    keys = []
    for k, v in env.items():
        if k.startswith("GROQ_API_KEY") and v.startswith("gsk_"):
            keys.append(v)
    return keys


def _load_log() -> List[Dict[str, Any]]:
    if not DOCTOR_LOG_FILE.exists():
        return []
    try:
        return json.loads(DOCTOR_LOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _append_log(entry: Dict[str, Any]):
    log_data = _load_log()
    log_data.append(entry)
    DOCTOR_LOG_FILE.write_text(
        json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _resolve_crash_file(crash_text: str) -> Optional[str]:
    """Extract the most relevant TYRON source file from a Python traceback."""
    matches = re.findall(r'File "([^"]+)", line \d+', crash_text)
    for match in reversed(matches):
        p = Path(match)
        try:
            rel = p.relative_to(BASE_DIR)
            rel_str = str(rel).replace("\\", "/")
            if rel_str.endswith(".py") and "venv" not in rel_str and "__pycache__" not in rel_str:
                # Never patch these critical files
                if rel_str not in {"run.py", "config.py", "app/main.py", "doctor.py"}:
                    return rel_str
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# AI FIXER
# ─────────────────────────────────────────────────────────────────────────────

class DoctorAIFixer:
    """Calls Groq API to generate a Python fix for a crash."""

    def __init__(self, groq_keys: List[str]):
        self._keys   = list(groq_keys)
        self._idx    = 0

    def _next_key(self) -> Optional[str]:
        if not self._keys:
            return None
        key = self._keys[self._idx % len(self._keys)]
        self._idx += 1
        return key

    def generate_fix(
        self,
        crash_text: str,
        file_content: str,
        affected_file: str,
        attempt: int = 1,
    ) -> Optional[str]:
        import urllib.request, urllib.error

        key = self._next_key()
        if not key:
            log.warning("Doctor: No Groq API keys found — cannot generate fix.")
            return None

        prompt = (
            "You are TYRON Doctor, an autonomous self-healing AI.\n"
            "TYRON's backend has CRASHED. Your job is to fix the broken Python file.\n\n"
            f"=== CRASH OUTPUT ===\n{crash_text[:4000]}\n\n"
            f"=== BROKEN FILE: {affected_file} ===\n```python\n{file_content[:5000]}\n```\n\n"
            f"Fix attempt #{attempt}.\n\n"
            "RULES:\n"
            "1. Find the root cause of the crash.\n"
            "2. Write the COMPLETE fixed Python file.\n"
            "3. Make the MINIMAL change — do not refactor unrelated code.\n"
            "4. Mark your fix with: # Doctor fix\n"
            "5. Output ONLY raw Python code — no markdown fences, no explanation.\n"
            "6. The fixed code must be importable without crashing.\n"
        )

        from config import GROQ_MODEL
        payload = json.dumps({
            "model"      : GROQ_MODEL or "openai/gpt-oss-120b",
            "messages"   : [
                {"role": "system", "content": "You are TYRON Doctor. Output only valid Python code."},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.15,
            "max_tokens" : 4096,
        }).encode("utf-8")

        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data    = payload,
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type" : "application/json",
            },
            method = "POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=50) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            code = data["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if LLM included them
            code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
            code = re.sub(r"\n?```$", "", code)
            return code.strip() or None
        except urllib.error.HTTPError as e:
            log.warning(f"Doctor AI HTTP error {e.code}: {e.reason}")
            return None
        except Exception as e:
            log.warning(f"Doctor AI request failed (attempt {attempt}): {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# SANDBOX TESTER
# ─────────────────────────────────────────────────────────────────────────────

def sandbox_test(fix_code: str, python_exe: str) -> Dict[str, Any]:
    """Run fix_code in an isolated subprocess. Returns pass/fail + output."""
    import uuid
    test_file = SANDBOX_DIR / f"doctor_test_{uuid.uuid4().hex[:8]}.py"
    try:
        test_file.write_text(fix_code, encoding="utf-8")
        proc = subprocess.Popen(
            [python_exe, "-u", str(test_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(BASE_DIR),
        )
        try:
            out, err = proc.communicate(timeout=SANDBOX_TIMEOUT)
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            rc = 124
            timed_out = True

        return {
            "passed"    : rc == 0 and not timed_out,
            "exit_code" : rc,
            "stdout"    : out.decode("utf-8", errors="replace")[:2000],
            "stderr"    : err.decode("utf-8", errors="replace")[:2000],
            "timed_out" : timed_out,
        }
    except Exception as e:
        return {"passed": False, "exit_code": -1, "stdout": "", "stderr": str(e), "timed_out": False}
    finally:
        try:
            test_file.unlink(missing_ok=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# AUTO PATCHER
# ─────────────────────────────────────────────────────────────────────────────

def apply_fix(relative_path: str, fix_code: str) -> Dict[str, Any]:
    """Backup original file and write fix_code in its place."""
    target = BASE_DIR / relative_path
    if not target.exists():
        return {"success": False, "message": f"Target not found: {relative_path}"}

    ts  = datetime.now(PKT).strftime("%Y%m%d_%H%M%S")
    bak = BACKUP_DIR / f"{target.stem}_backup_{ts}.py"

    try:
        shutil.copy2(str(target), str(bak))
        target.write_text(fix_code, encoding="utf-8")
        return {
            "success"     : True,
            "backup_path" : str(bak),
            "message"     : f"Fix applied to '{relative_path}'. Backup: '{bak.name}'.",
        }
    except Exception as e:
        return {"success": False, "message": f"Patch error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# DOCTOR — core watchdog loop
# ─────────────────────────────────────────────────────────────────────────────

class TyronDoctor:
    """
    Runs TYRON as a subprocess and heals it whenever it crashes.
    Doctor itself never crashes — every risky operation is wrapped in try/except.
    """

    def __init__(self):
        _ensure_dirs()
        self._env         = _load_env()
        self._groq_keys   = _get_groq_keys(self._env)
        self._fixer       = DoctorAIFixer(self._groq_keys)
        self._python_exe  = sys.executable
        self._tyron_proc: Optional[subprocess.Popen] = None
        self._shutdown    = threading.Event()
        self._restart_count = 0
        self._backoff     = RESTART_BACKOFF_BASE

        log.info("═" * 62)
        log.info("  TYRON DOCTOR  —  External Self-Healing Watchdog")
        log.info("═" * 62)
        log.info(f"  Base dir    : {BASE_DIR}")
        log.info(f"  Groq keys   : {len(self._groq_keys)} key(s) loaded")
        log.info(f"  Python exe  : {self._python_exe}")
        log.info(f"  Doctor log  : {DOCTOR_LOG_FILE}")
        log.info("═" * 62)

        if not self._groq_keys:
            log.warning("No GROQ_API_KEY found in .env — Doctor can watch but cannot auto-fix.")

        # Handle Ctrl+C gracefully
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info("Doctor received shutdown signal — stopping TYRON and exiting...")
        self._shutdown.set()
        self._kill_tyron()
        sys.exit(0)

    def _kill_tyron(self):
        if self._tyron_proc and self._tyron_proc.poll() is None:
            try:
                self._tyron_proc.terminate()
                self._tyron_proc.wait(timeout=5)
            except Exception:
                try:
                    self._tyron_proc.kill()
                except Exception:
                    pass

    def _start_tyron(self) -> subprocess.Popen:
        """Launch TYRON as a subprocess, capturing all output."""
        log.info(f"🚀 Starting TYRON (attempt #{self._restart_count + 1})...")

        # Force UTF-8 encoding so Unicode chars (box-drawing, emoji) never crash TYRON
        tyron_env = {
            **os.environ,
            **self._env,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8"      : "1",
        }

        proc = subprocess.Popen(
            [self._python_exe, "-u", "run.py"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,   # merge stderr → stdout for one stream
            cwd    = str(BASE_DIR),
            env    = tyron_env,
            bufsize= 0,                   # unbuffered
        )
        log.info(f"   TYRON PID: {proc.pid}")
        return proc

    def _stream_output(self, proc: subprocess.Popen, buffer: List[str]):
        """
        Reads TYRON's output line by line in a thread.
        Prints it to console AND stores last N lines in buffer for crash analysis.
        """
        try:
            for raw_line in proc.stdout:
                if self._shutdown.is_set():
                    break
                try:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                except Exception:
                    line = repr(raw_line)
                print(f"  [TYRON] {line}", flush=True)
                buffer.append(line)
                if len(buffer) > CRASH_BUFFER_LINES:
                    buffer.pop(0)
        except Exception:
            pass

    def _heal_crash(self, crash_text: str) -> bool:
        """
        Try to fix the crash. Returns True if fix was applied successfully.
        """
        log.info("🩺 Doctor analyzing crash...")

        affected_file = _resolve_crash_file(crash_text)
        if not affected_file:
            log.warning("Doctor: No patchable file identified in crash. Will restart TYRON anyway.")
            return False

        log.info(f"   Affected file: {affected_file}")

        target = BASE_DIR / affected_file
        try:
            file_content = target.read_text(encoding="utf-8")
        except Exception as e:
            log.warning(f"Doctor: Cannot read {affected_file}: {e}")
            return False

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            log.info(f"   🔧 Fix attempt {attempt}/{MAX_FIX_ATTEMPTS} — calling Groq AI...")
            fix_code = self._fixer.generate_fix(
                crash_text    = crash_text,
                file_content  = file_content,
                affected_file = affected_file,
                attempt       = attempt,
            )

            if not fix_code:
                log.info(f"   AI returned no fix on attempt {attempt}.")
                continue

            log.info(f"   🧪 Sandbox testing fix (attempt {attempt})...")
            result = sandbox_test(fix_code, self._python_exe)

            if result["passed"]:
                log.info(f"   ✅ Sandbox PASSED — applying fix to '{affected_file}'...")
                patch = apply_fix(affected_file, fix_code)

                if patch["success"]:
                    log.info(f"   ✅ Fix applied! {patch['message']}")
                    _append_log({
                        "timestamp"    : _now_iso(),
                        "event"        : "FIX_APPLIED",
                        "affected_file": affected_file,
                        "fix_attempt"  : attempt,
                        "backup"       : patch["backup_path"],
                        "crash_snippet": crash_text[-800:],
                    })
                    return True
                else:
                    log.warning(f"   Patch apply failed: {patch['message']}")
                    return False
            else:
                log.info(f"   Sandbox FAILED (attempt {attempt}). stderr: {result['stderr'][:150]}")

        log.warning(f"   ❌ All {MAX_FIX_ATTEMPTS} fix attempts failed. Restarting without fix.")
        _append_log({
            "timestamp"    : _now_iso(),
            "event"        : "FIX_FAILED",
            "affected_file": affected_file,
            "crash_snippet": crash_text[-800:],
        })
        return False

    def run(self):
        """Main watchdog loop — runs forever until Ctrl+C."""
        while not self._shutdown.is_set():
            if self._restart_count >= MAX_RESTART_ATTEMPTS:
                log.error(
                    f"Doctor: TYRON has crashed {MAX_RESTART_ATTEMPTS} times. "
                    "Giving up. Please review the crashes manually."
                )
                _append_log({"timestamp": _now_iso(), "event": "MAX_RESTARTS_REACHED"})
                break

            output_buffer: List[str] = []
            start_time = time.monotonic()

            try:
                proc = self._start_tyron()
                self._tyron_proc = proc

                # Stream output in background thread
                stream_thread = threading.Thread(
                    target=self._stream_output,
                    args=(proc, output_buffer),
                    daemon=True,
                )
                stream_thread.start()

                # Wait for TYRON to exit
                exit_code = proc.wait()
                stream_thread.join(timeout=3)

                uptime = time.monotonic() - start_time

            except Exception as e:
                log.error(f"Doctor: Failed to start TYRON: {e}")
                exit_code = -1
                uptime    = 0

            if self._shutdown.is_set():
                break

            # ── Classify exit ──────────────────────────────────────────────────
            if exit_code == 0:
                log.info("TYRON exited cleanly (code 0). Doctor exiting too.")
                break

            # It's a crash
            self._restart_count += 1
            uptime_str = f"{uptime:.1f}s"
            log.warning(
                f"\n{'═'*60}\n"
                f"  💥 TYRON CRASHED  (exit code {exit_code}, uptime {uptime_str})\n"
                f"{'═'*60}"
            )

            # Save crash to file
            crash_text = "\n".join(output_buffer)
            try:
                CRASH_LOG_FILE.write_text(crash_text, encoding="utf-8")
                log.info(f"   Crash saved to: {CRASH_LOG_FILE}")
            except Exception:
                pass

            _append_log({
                "timestamp"   : _now_iso(),
                "event"       : "CRASH",
                "exit_code"   : exit_code,
                "uptime_sec"  : round(uptime, 1),
                "restart_num" : self._restart_count,
            })

            # If TYRON ran healthy for a while, reset backoff
            if uptime > MIN_UPTIME_HEALTHY:
                self._backoff = RESTART_BACKOFF_BASE
                log.info(f"   TYRON ran for {uptime_str} → backoff reset.")

        # ── Try to heal using Guardian Repair Engine ─────────────────────────────────
            try:
                from guardian.repair_engine import RepairEngine
                engine = RepairEngine()
                inc_id = f"legacy_doctor_{datetime.now(PKT).strftime('%Y%m%d_%H%M%S')}"
                repair_res = engine.propose_and_apply_repair(
                    incident_id=inc_id,
                    traceback_text=crash_text,
                    recent_logs=output_buffer,
                    dry_run=False,
                )

                if not repair_res.get("success"):
                    log.error(f"❌ Doctor Repair Failed ({repair_res.get('reason')}). Stopping restart to prevent crash loop.")
                    log.info("ℹ️ Recommended action: Use 'python tyron_guardian.py' for full independent supervisor management.")
                    _append_log({"timestamp": _now_iso(), "event": "REPAIR_FAILED_STOP", "reason": repair_res.get("reason")})
                    break

                log.info(f"✅ Fix validated and applied via {repair_res.get('patch_level')}. Restarting TYRON...")
            except Exception as heal_err:
                log.error(f"Doctor heal raised exception: {heal_err}. Stopping to prevent blind restart loop.")
                break

            # Backoff before restart
            log.info(f"   ⏱  Waiting {self._backoff}s before restarting TYRON...")
            self._shutdown.wait(timeout=self._backoff)
            self._backoff = min(self._backoff * 2, MAX_BACKOFF)

            log.info(f"\n{'─'*60}\n  Restart #{self._restart_count + 1}\n{'─'*60}")

        log.info("Doctor shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print(" [DOCTOR.PY]  LEGACY DOCTOR COMPATIBILITY WRAPPER")
    print(" NOTICE: For full independent supervision and multi-level repair, use:")
    print("         python tyron_guardian.py")
    print("=" * 70)
    doctor = TyronDoctor()
    doctor.run()

