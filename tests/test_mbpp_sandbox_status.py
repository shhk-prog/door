import pytest

from lineage_collusion.benchmarks.code_sandbox import probe_sandbox
from lineage_collusion.benchmarks.reasoning_adapters import evaluate_mbpp_code_execution

pytestmark = pytest.mark.skipif(not probe_sandbox()["available"], reason="bwrap サンドボックスが使えない環境")

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

def test_mbpp_sandbox_dangerous_code_has_no_host_effect(tmp_path):
    """
    OS レベル隔離（bwrap）では危険な操作を「検出」するのではなく「ホストに影響させない」。
    サンドボックス内からホストのディレクトリへの書き込みもネットワーク接続もできないことを確認する。
    （旧実装の静的禁止語チェックは、正当なコードを誤って違反扱いし得るため廃止した）
    """
    marker = tmp_path / "hacked_marker"
    code = f"""
import os
os.system("touch {marker}")
def add(a, b): return a + b
"""
    res = evaluate_mbpp_code_execution(code, ["assert add(1, 2) == 3"], timeout=5.0)
    assert not marker.exists(), "サンドボックス内の操作がホストのファイルシステムに影響した"
    assert res["status"] in ("passed", "runtime_error", "sandbox_violation")

    net = evaluate_mbpp_code_execution(
        "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\ndef f(): return 1",
        ["assert f() == 1"], timeout=5.0)
    assert net["passed"] is False and net["status"] == "sandbox_violation"


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
