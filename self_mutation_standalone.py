"""
self_mutation.py — Confirmation-Gated Self-Mutation Pattern
================================================================

A minimal, dependency-free reference implementation of a safety pattern
for AI agents that modify their own source code:

    propose -> sandbox test -> human approval -> backup -> apply
                                                       |
                                                   rollback (anytime)

See SELF_MUTATION_WHITEPAPER.md in this repo for the full write-up of
why this pattern exists and what it does / doesn't protect against.

USAGE:

    evolution = SelfMutation(project_root=".", mutation_dir="mutations")

    # Step 1: an LLM (or anything else) proposes a full replacement file
    evolution.write_patch(new_code, description="Add retry logic")

    # Step 2: test it in isolation before it goes anywhere near production
    result = evolution.test_patch()
    if not result["passed"]:
        print("Patch failed:", result["stderr"])
        exit()

    # Step 3: a human reviews result + evolution.preview_patch(), then:
    outcome = evolution.apply_patch("my_module.py", description="...")
    print(outcome["message"])

    # Anytime later, if the change turns out to be wrong:
    evolution.rollback("my_module.py")

No dependencies beyond the Python standard library.
"""

import json
import subprocess
import sys
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


class SelfMutation:
    """
    Manages the propose -> test -> apply -> rollback lifecycle for
    self-generated code patches.

    Args:
        project_root: root directory that target files are relative to.
        mutation_dir: subdirectory (relative to project_root) used to
                      store the pending patch, backups, and the change log.
        test_timeout: max seconds to let the sandbox test run before
                      killing it and treating the patch as failed.
    """

    def __init__(
        self,
        project_root: str = ".",
        mutation_dir: str = ".mutations",
        test_timeout: int = 15,
    ):
        self.project_root = Path(project_root)
        self.mutation_dir = self.project_root / mutation_dir
        self.backup_dir = self.mutation_dir / "backups"
        self.temp_patch = self.mutation_dir / "temp_patch.py"
        self.log_file = self.mutation_dir / "mutation_log.json"
        self.test_timeout = test_timeout

        self.mutation_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1 — PROPOSE
    # =========================================================================

    def write_patch(self, code: str, description: str = "") -> Dict[str, Any]:
        """
        Write a proposed full-file replacement to the pending patch slot.
        This does NOT touch any real file yet — it's just staged.
        """
        code = code.strip()
        self.temp_patch.write_text(code, encoding="utf-8")
        return {
            "path": str(self.temp_patch),
            "size_bytes": len(code.encode("utf-8")),
            "description": description,
            "written_at": datetime.now().isoformat(),
        }

    def preview_patch(self) -> Dict[str, Any]:
        """Return the currently staged patch's content for human review."""
        if not self.temp_patch.exists():
            return {"exists": False, "content": "", "size_bytes": 0}
        content = self.temp_patch.read_text(encoding="utf-8")
        return {
            "exists": True,
            "content": content,
            "size_bytes": len(content.encode("utf-8")),
        }

    # =========================================================================
    # STEP 2 — SANDBOX TEST
    # =========================================================================

    def test_patch(self) -> Dict[str, Any]:
        """
        Run the staged patch as a standalone subprocess with a timeout.
        Catches syntax errors, import failures, and immediate crashes.

        This is NOT a security sandbox against adversarial code — it's a
        correctness check against an LLM (or other generator) making an
        honest mistake. See the whitepaper's Limitations section.
        """
        if not self.temp_patch.exists():
            return {
                "passed": False, "exit_code": -1, "stdout": "",
                "stderr": "No patch staged. Call write_patch() first.",
                "timed_out": False, "duration_ms": 0,
            }

        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(self.temp_patch)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = proc.communicate(timeout=self.test_timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout_b, stderr_b = proc.communicate()
                exit_code = 124
                timed_out = True
        except Exception as e:
            return {
                "passed": False, "exit_code": -1, "stdout": "",
                "stderr": f"Internal error: {e}", "timed_out": False,
                "duration_ms": int((time.monotonic() - start) * 1000),
            }

        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "passed": exit_code == 0 and not timed_out,
            "exit_code": exit_code,
            "stdout": stdout_b.decode("utf-8", errors="replace")[:10_000],
            "stderr": stderr_b.decode("utf-8", errors="replace")[:10_000],
            "timed_out": timed_out,
            "duration_ms": duration_ms,
        }

    # =========================================================================
    # STEP 3 & 4 — HUMAN APPROVAL (external) + BACKUP + APPLY
    # =========================================================================
    # Note: "human approval" is not a method call here — it's whatever gate
    # your application puts in front of calling apply_patch(). This library
    # doesn't enforce it; your application does, by only calling apply_patch
    # after a human has reviewed preview_patch() + test_patch() results.

    def apply_patch(self, target_relative_path: str, description: str = "") -> Dict[str, Any]:
        """
        Back up the target file, then replace it with the staged patch.
        Only call this AFTER a human has approved the change — this
        library provides the mechanism, not the policy.
        """
        if not self.temp_patch.exists():
            return {"success": False, "message": "No patch staged."}

        target_path = self.project_root / target_relative_path
        if not target_path.exists():
            return {"success": False, "message": f"Target not found: {target_relative_path}"}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{target_path.stem}_backup_{ts}.py"
        backup_path = self.backup_dir / backup_name

        try:
            shutil.copy2(str(target_path), str(backup_path))
            patch_content = self.temp_patch.read_text(encoding="utf-8")
            target_path.write_text(patch_content, encoding="utf-8")
            self.temp_patch.unlink(missing_ok=True)

            self._append_log({
                "applied_at": datetime.now().isoformat(),
                "target": target_relative_path,
                "backup": str(backup_path),
                "description": description,
            })

            return {
                "success": True,
                "target_path": str(target_path),
                "backup_path": str(backup_path),
                "message": f"Applied. Backup at '{backup_name}'.",
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to apply: {e}"}

    # =========================================================================
    # ROLLBACK
    # =========================================================================

    def rollback(self, target_relative_path: str) -> Dict[str, Any]:
        """Restore the most recent backup for a given target file."""
        stem = Path(target_relative_path).stem
        backups = sorted(self.backup_dir.glob(f"{stem}_backup_*.py"), reverse=True)
        if not backups:
            return {"success": False, "message": f"No backups found for '{target_relative_path}'."}

        latest = backups[0]
        target_path = self.project_root / target_relative_path
        try:
            shutil.copy2(str(latest), str(target_path))
            return {
                "success": True,
                "restored_from": str(latest),
                "message": f"Restored from '{latest.name}'.",
            }
        except Exception as e:
            return {"success": False, "message": f"Rollback failed: {e}"}

    # =========================================================================
    # HISTORY
    # =========================================================================

    def get_history(self) -> List[Dict[str, Any]]:
        """Return all applied patches, most recent first."""
        return list(reversed(self._load_log()))

    def _load_log(self) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        try:
            return json.loads(self.log_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _append_log(self, entry: Dict[str, Any]):
        log = self._load_log()
        log.append(entry)
        self.log_file.write_text(json.dumps(log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Minimal demo — proposes a patch to itself... well, to a demo file.
    demo_target = Path("demo_target.py")
    demo_target.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")

    evo = SelfMutation(project_root=".")

    new_code = "def greet():\n    return 'hello, improved!'\n"
    print(evo.write_patch(new_code, description="Improve greeting"))
    print(evo.test_patch())
    print(evo.preview_patch())
    # In a real application, a human reviews the above two results before
    # this next line ever runs:
    print(evo.apply_patch("demo_target.py", description="Improve greeting"))
    print(evo.get_history())
