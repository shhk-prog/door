"""
生成コード（MBPP）の OS レベル隔離実行。

研究サーバー上で生成コードを直接実行しない。bubblewrap（bwrap）で次を強制する:
  - 名前空間分離: --unshare-all（ネットワーク・PID・IPC・UTS・user・cgroup）、--die-with-parent、--new-session
  - ファイルシステム: システムディレクトリ（/usr 等）を読み取り専用でのみ公開。ホーム・NAS・作業ディレクトリは不可視。
    /tmp は空の tmpfs を読み取り専用に再マウント（書き込み不可）
  - 環境変数: --clearenv（PATH のみ再設定）
  - rlimit: アドレス空間・CPU 時間・ファイルサイズ 0・コアダンプ 0
  - インタプリタ: システムの Python を -I -S -B で起動（venv・site-packages を読まない）

判定はサンドボックス内ランナーの完了報告（実行ごとの乱数ノンス付き JSON 行）で行う。生成コードが
os._exit(0) などで早期終了しても、ノンス付きの "passed" 報告が無ければ合格にならない。
状態: passed / syntax_error / assertion_failed / runtime_error / timeout / sandbox_violation /
      empty_code / environment_error（隔離環境の起動失敗など）。
限界: 同一プロセス内でランナーのフレームを探索するような敵対的コードは想定外（完全なコンテナ隔離でもない）。
"""

import json
import os
import secrets
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

SANDBOX_STATUSES = ("passed", "syntax_error", "assertion_failed", "runtime_error", "timeout",
                    "sandbox_violation", "empty_code", "environment_error")

# サンドボックス内で実行するランナー（標準入力から JSON を受け取り、標準入力を閉じてから実行する）
_RUNNER = r'''
import sys, json, os, errno
payload = json.loads(sys.stdin.read())
sys.stdin.close()
nonce = payload.pop("nonce")
def _report(status, msg=""):
    sys.stdout.flush()
    sys.stdout.write("\n" + json.dumps({"nonce": nonce, "status": status, "msg": str(msg)[:300]}) + "\n")
    sys.stdout.flush()
    os._exit(0)
def _classify(e):
    if isinstance(e, PermissionError) or (isinstance(e, OSError) and getattr(e, "errno", None) in (errno.EROFS, errno.EACCES, errno.EPERM, errno.ENETUNREACH, errno.EAFNOSUPPORT)):
        return "sandbox_violation"
    return "runtime_error"
ns = {"__name__": "__main__"}
try:
    code_obj = compile(payload["code"], "<generated>", "exec")
except (SyntaxError, ValueError) as e:
    _report("syntax_error", repr(e))
try:
    exec(code_obj, ns)
except BaseException as e:
    _report(_classify(e), repr(e))
# MBPP の test_setup_code は解答で定義されたクラス等を参照するため解答の後に実行する（例: task 367, 927）
try:
    if payload.get("setup"):
        exec(compile(payload["setup"], "<setup>", "exec"), ns)
except BaseException as e:
    _report(_classify(e), "setup: " + repr(e))
for t in payload["tests"]:
    try:
        exec(compile(t, "<test>", "exec"), ns)
    except AssertionError as e:
        _report("assertion_failed", t)
    except BaseException as e:
        _report(_classify(e), repr(e))
_report("passed")
'''

# 読み取り専用で公開するシステムディレクトリ（存在するもののみ）
_SYSTEM_RO_DIRS = ("/usr", "/lib", "/lib64", "/bin", "/etc/alternatives")


class SandboxUnavailableError(RuntimeError):
    """OS レベル隔離実行環境が利用できない（評価を停止する）。"""


def _bwrap_cmd(python: str) -> List[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailableError("bubblewrap (bwrap) が見つかりません")
    cmd = [bwrap, "--unshare-all", "--die-with-parent", "--new-session"]
    for d in _SYSTEM_RO_DIRS:
        if os.path.lexists(d):
            if os.path.islink(d):
                cmd += ["--symlink", os.readlink(d), d]
            else:
                cmd += ["--ro-bind", d, d]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--remount-ro", "/tmp",
            "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONHASHSEED", "0",
            python, "-I", "-S", "-B", "-c", _RUNNER]
    return cmd


def _limits(mem_mb: int, cpu_s: int):
    def _apply():
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return _apply


def run_in_sandbox(code: str, tests: Sequence[str], setup: str = "", timeout: float = 10.0,
                   python: str = "/usr/bin/python3", mem_mb: int = 1536) -> Dict[str, Any]:
    """生成コードを隔離環境で実行して状態を返す。隔離環境自体が無い場合は SandboxUnavailableError。"""
    if not os.path.exists(python):
        raise SandboxUnavailableError(f"サンドボックス用 Python {python} がありません")
    cmd = _bwrap_cmd(python)
    nonce = secrets.token_hex(16)
    payload = json.dumps({"nonce": nonce, "code": code, "setup": setup or "", "tests": list(tests)})
    try:
        proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, timeout=timeout,
                              env={}, preexec_fn=_limits(mem_mb, max(2, int(timeout) + 1)))
    except subprocess.TimeoutExpired:
        return {"passed": False, "status": "timeout", "error_message": f"wall-clock timeout {timeout}s", "exit_code": None}
    except OSError as e:
        raise SandboxUnavailableError(f"bwrap の起動に失敗: {e}") from e

    report = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("nonce") == nonce:
            report = obj
            break
    if report is None:
        stderr = (proc.stderr or "").strip()
        if proc.returncode is not None and proc.returncode < 0:
            import signal
            sig = -proc.returncode
            status = "timeout" if sig in (signal.SIGXCPU, signal.SIGKILL) else "runtime_error"
            return {"passed": False, "status": status, "error_message": f"killed by signal {sig}",
                    "exit_code": proc.returncode}
        if "bwrap:" in stderr:
            # 隔離環境の起動そのものの失敗（コード起因ではない）
            return {"passed": False, "status": "environment_error", "error_message": stderr[-300:],
                    "exit_code": proc.returncode}
        return {"passed": False, "status": "runtime_error",
                "error_message": "no completion report (abnormal exit) " + stderr[-200:], "exit_code": proc.returncode}
    status = report["status"]
    if status not in SANDBOX_STATUSES:
        status = "runtime_error"
    return {"passed": status == "passed", "status": status, "error_message": report.get("msg", ""),
            "exit_code": proc.returncode}


@lru_cache(maxsize=None)
def probe_sandbox(python: str = "/usr/bin/python3") -> Dict[str, Any]:
    """
    隔離が実際に効いているかを確認する（ネットワーク遮断・書き込み不可・ホーム不可視・正常コードが通る）。
    1 つでも満たさなければ available=False。
    """
    checks = {}
    try:
        ok = run_in_sandbox("def f(a, b):\n    return a + b\n", ["assert f(1, 2) == 3"], python=python)
        checks["valid_code_passes"] = ok["status"] == "passed"
        net = run_in_sandbox("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\n", [], python=python)
        checks["network_blocked"] = net["status"] in ("sandbox_violation", "runtime_error")
        wr = run_in_sandbox("open('/tmp/_probe', 'w').write('x')\n", [], python=python)
        checks["write_blocked"] = wr["status"] == "sandbox_violation"
        home = os.path.expanduser("~")
        hv = run_in_sandbox(f"import os\nassert not os.path.exists({home!r})\n", [], python=python)
        checks["home_invisible"] = hv["status"] == "passed"
    except SandboxUnavailableError as e:
        return {"available": False, "reason": str(e), "checks": checks}
    available = all(checks.values())
    return {"available": available, "reason": None if available else "isolation checks failed", "checks": checks,
            "python": python, "bwrap": shutil.which("bwrap")}
