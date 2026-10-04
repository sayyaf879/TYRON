"""
CODE SANDBOX MODELS
====================

PURPOSE:
  Request / response models for the POST /code/run endpoint.

FIELDS:
  CodeRunRequest:
    code       - Python source code to execute (max 10,000 chars)
    timeout    - max seconds to run before killing (1-30, default 10)
    stdin      - optional string piped to the process as standard input

  CodeRunResponse:
    stdout     - everything the code printed to stdout
    stderr     - compile errors, tracebacks, warnings
    exit_code  - 0 = success, 1 = error, 124 = timeout, -1 = internal error
    timed_out  - True if the process was killed for exceeding timeout
    duration_ms- wall-clock execution time in milliseconds
"""

from pydantic import BaseModel, Field
from typing import Optional


class CodeRunRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=10_000,
                      description="Python source code to execute")
    timeout: int = Field(default=10, ge=1, le=30,
                         description="Max execution time in seconds (1-30)")
    stdin: Optional[str] = Field(default=None, max_length=5_000,
                                 description="Optional stdin to pipe to the process")


class CodeRunResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int