"""
Sandboxed code execution with timeout.

Executes generated code against unit tests in an isolated subprocess.
Returns structured results including pass/fail, classified error type, stderr.

Error Taxonomy (matching prompt specification):
  - syntax_error:       SyntaxError, IndentationError, TabError
  - runtime_type:       TypeError, AttributeError, NameError
  - runtime_value:      ValueError, KeyError, IndexError, ZeroDivisionError
  - runtime_resource:   RecursionError, MemoryError, ImportError, OverflowError
  - logical_error:      AssertionError (wrong output — tests fail)
  - timeout:            Execution exceeded time limit
"""

import subprocess
import sys
import tempfile
import os
import time
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class ErrorType(Enum):
    """
    Granular error categories matching the prompt taxonomy.
    
    Aggregated groups for reporting:
      SYNTAX  = syntax_error
      RUNTIME = runtime_type + runtime_value + runtime_resource
      LOGICAL = logical_error
      TIMEOUT = timeout
    """
    NONE = "none"
    SYNTAX = "syntax_error"
    RUNTIME_TYPE = "runtime_type"
    RUNTIME_VALUE = "runtime_value"
    RUNTIME_RESOURCE = "runtime_resource"
    LOGICAL = "logical_error"
    TIMEOUT = "timeout"

    @property
    def aggregated_group(self) -> str:
        """Return the aggregated group name for this error type."""
        _groups = {
            "none": "NONE",
            "syntax_error": "SYNTAX",
            "runtime_type": "RUNTIME",
            "runtime_value": "RUNTIME",
            "runtime_resource": "RUNTIME",
            "logical_error": "LOGICAL",
            "timeout": "TIMEOUT",
        }
        return _groups.get(self.value, "RUNTIME")


# Keep backward compatibility alias
AGGREGATED_GROUPS = {
    "SYNTAX": ["syntax_error"],
    "RUNTIME": ["runtime_type", "runtime_value", "runtime_resource"],
    "LOGICAL": ["logical_error"],
    "TIMEOUT": ["timeout"],
}


@dataclass
class ExecutionResult:
    """Result of executing a code candidate against tests."""
    passed: bool = False
    error_type: ErrorType = ErrorType.NONE
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    
    def to_dict(self):
        return {
            "passed": self.passed,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
        }


def _classify_error(stderr: str, returncode: int) -> ErrorType:
    """
    Classify error type from stderr output using the granular taxonomy.
    
    Priority order matters: check more specific types first.
    """
    if returncode == 0:
        return ErrorType.NONE
    
    stderr_lower = stderr.lower()
    
    # ── Syntax errors ──
    if any(kw in stderr_lower for kw in [
        "syntaxerror", "indentationerror", "taberror"
    ]):
        return ErrorType.SYNTAX
    
    # ── Logical errors (assertion failures = wrong output) ──
    if "assertionerror" in stderr_lower:
        return ErrorType.LOGICAL
    
    # ── Runtime: Type-related ──
    if any(kw in stderr_lower for kw in [
        "typeerror", "attributeerror", "nameerror"
    ]):
        return ErrorType.RUNTIME_TYPE
    
    # ── Runtime: Value-related ──
    if any(kw in stderr_lower for kw in [
        "valueerror", "keyerror", "indexerror", "zerodivisionerror"
    ]):
        return ErrorType.RUNTIME_VALUE
    
    # ── Runtime: Resource-related ──
    if any(kw in stderr_lower for kw in [
        "recursionerror", "memoryerror", "importerror",
        "overflowerror", "modulenotfounderror"
    ]):
        return ErrorType.RUNTIME_RESOURCE
    
    # ── Timeout ──
    if "timeout" in stderr_lower or returncode == -9:
        return ErrorType.TIMEOUT
    
    # ── Default: runtime_type (safest default) ──
    return ErrorType.RUNTIME_TYPE


def _count_tests_in_code(test_code: str) -> int:
    """Estimate number of test assertions in test code."""
    count = 0
    for line in test_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            count += 1
    return max(count, 1)  # at least 1


def execute_code(
    code: str,
    test_code: str,
    entry_point: str = "",
    timeout: float = 5.0,
) -> ExecutionResult:
    """
    Execute generated code against test cases in a sandboxed subprocess.
    
    Args:
        code: The generated code (function definition)
        test_code: The test code (assertions / check function call)
        entry_point: The function name being tested
        timeout: Maximum execution time in seconds
    
    Returns:
        ExecutionResult with pass/fail, error classification, timing
    """
    # Build the full program: code + tests
    full_program = f"{code}\n\n{test_code}\n"
    
    total_tests = _count_tests_in_code(test_code)
    
    # Write to temp file
    tmp_dir = os.path.join(os.path.dirname(__file__), ".tmp_exec")
    os.makedirs(tmp_dir, exist_ok=True)
    
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=tmp_dir, delete=False, encoding="utf-8"
    )
    tmp_file.write(full_program)
    tmp_file.close()
    
    result = ExecutionResult(tests_total=total_tests)
    
    try:
        start_time = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, tmp_file.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        elapsed = time.perf_counter() - start_time
        
        result.execution_time = elapsed
        result.stdout = proc.stdout[:2000]  # cap output
        result.stderr = proc.stderr[:2000]
        
        if proc.returncode == 0:
            result.passed = True
            result.tests_passed = total_tests
            result.error_type = ErrorType.NONE
        else:
            result.passed = False
            result.error_type = _classify_error(proc.stderr, proc.returncode)
            result.error_message = proc.stderr[:500]
            # Estimate partial test passes for logical errors
            if result.error_type == ErrorType.LOGICAL:
                passed_before = 0
                for line in proc.stderr.split("\n"):
                    if "assert" in line.lower() and "error" in line.lower():
                        break
                    passed_before += 1
                result.tests_passed = min(passed_before, total_tests - 1)
    
    except subprocess.TimeoutExpired:
        result.execution_time = timeout
        result.passed = False
        result.error_type = ErrorType.TIMEOUT
        result.error_message = f"Execution timed out after {timeout}s"
    
    except Exception as e:
        result.passed = False
        result.error_type = ErrorType.RUNTIME_TYPE
        result.error_message = str(e)
    
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass
    
    return result


def execute_code_batch(
    codes: List[str],
    test_code: str,
    entry_point: str = "",
    timeout: float = 5.0,
) -> List[ExecutionResult]:
    """Execute multiple code candidates against the same tests (sequential)."""
    return [
        execute_code(code, test_code, entry_point, timeout)
        for code in codes
    ]


def execute_code_batch_parallel(
    codes: List[str],
    test_code: str,
    entry_point: str = "",
    timeout: float = 5.0,
    max_workers: int = 4,
) -> List[ExecutionResult]:
    """
    Execute multiple code candidates in parallel using ThreadPoolExecutor.

    Each candidate runs in its own subprocess, so parallel execution is safe.
    Results are returned in the same order as the input codes.

    Args:
        codes: List of code strings to execute
        test_code: Test code to run against each candidate
        entry_point: Function name being tested
        timeout: Maximum execution time per candidate
        max_workers: Number of parallel workers

    Returns:
        List of ExecutionResult in same order as input
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not codes:
        return []

    # For small batches, sequential is fine
    if len(codes) <= 2:
        return execute_code_batch(codes, test_code, entry_point, timeout)

    results = [None] * len(codes)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(execute_code, code, test_code, entry_point, timeout): i
            for i, code in enumerate(codes)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = ExecutionResult(
                    passed=False,
                    error_type=ErrorType.RUNTIME_TYPE,
                    error_message=f"Parallel execution error: {e}",
                )

    return results

