import pytest
from lineage_collusion.benchmarks.reasoning_adapters import evaluate_mbpp_code_execution

def test_mbpp_sandbox_passed():
    code = """
def add(a, b):
    return a + b
"""
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    res = evaluate_mbpp_code_execution(code, tests, timeout=2.0)
    assert res["passed"] is True
    assert res["status"] == "passed"

def test_mbpp_sandbox_assertion_failed():
    code = """
def add(a, b):
    return a - b
"""
    tests = ["assert add(1, 2) == 3"]
    res = evaluate_mbpp_code_execution(code, tests, timeout=2.0)
    assert res["passed"] is False
    assert res["status"] == "assertion_failed"

def test_mbpp_sandbox_syntax_error():
    code = """
def add(a, b)
    return a + b
"""
    tests = ["assert add(1, 2) == 3"]
    res = evaluate_mbpp_code_execution(code, tests, timeout=2.0)
    assert res["passed"] is False
    assert res["status"] == "syntax_error"

def test_mbpp_sandbox_dangerous_code():
    code = """
import os
os.system("echo hacked")
def add(a, b): return a + b
"""
    tests = ["assert add(1, 2) == 3"]
    res = evaluate_mbpp_code_execution(code, tests, timeout=2.0)
    assert res["passed"] is False
    assert res["status"] == "sandbox_violation"

def test_mbpp_sandbox_file_write_violation():
    code = """
def add(a, b):
    with open('/tmp/test_write.txt', 'w') as f:
        f.write('hacked')
    return a + b
"""
    tests = ["assert add(1, 2) == 3"]
    res = evaluate_mbpp_code_execution(code, tests, timeout=2.0)
    assert res["passed"] is False
    assert res["status"] == "sandbox_violation"

def test_mbpp_sandbox_timeout():
    code = """
def add(a, b):
    while True:
        pass
    return a + b
"""
    tests = ["assert add(1, 2) == 3"]
    res = evaluate_mbpp_code_execution(code, tests, timeout=1.0)
    assert res["passed"] is False
    assert res["status"] == "timeout"
