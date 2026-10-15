"""
オンライン遠隔制御のネットワーク通信トランスポート（TCPソケット通信）。
提供者エージェント（Provider Agent）からの命令メッセージをローカル TCP ソケットを介して
対象エージェント（Target Agent）へ送信・受領確認（ACK）を行い、実通信送達を検証する。

注意（学術的境界）:
本実装は同一プロセス内のローカル・ループバック（127.0.0.1）による送受信（プロセス内 IPC / 通信プロトコル検証）
であり、独立した別ホストまたは分離された物理エージェント間通信ではありません。
過大な主張（overclaim）を防ぐため、テレメトリに scope: same_process_loopback_ipc を明記します。
"""

import json
import socket
import time
from typing import Any, Dict, Optional, Tuple


def transmit_command_via_socket(
    payload: Dict[str, Any],
    host: str = "127.0.0.1",
    timeout: float = 2.0,
) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    ローカル TCP ソケット経由で命令メッセージを送受信する。

    戻り値:
        (is_delivered, received_payload, telemetry_meta)
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ポート 0 で OS に動的空きポートを割り当てさせる
    server.bind((host, 0))
    port = server.getsockname()[1]
    server.listen(1)
    server.settimeout(timeout)

    t0 = time.perf_counter()
    received_payload = None
    bytes_sent = 0
    bytes_received = 0

    try:
        # クライアント側（Provider）
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(timeout)
        client.connect((host, port))

        raw_data = json.dumps(payload).encode("utf-8")
        client.sendall(raw_data)
        bytes_sent = len(raw_data)

        # サーバー側（Target）
        conn, _ = server.accept()
        conn.settimeout(timeout)
        data = conn.recv(65536)
        bytes_received = len(data)
        if data:
            received_payload = json.loads(data.decode("utf-8"))
            conn.sendall(b'{"status":"ACK"}')

        ack = client.recv(1024)
        client.close()
        conn.close()
        t1 = time.perf_counter()
        rtt_ms = round((t1 - t0) * 1000.0, 3)

        meta = {
            "transport": "tcp_socket",
            "scope": "same_process_loopback_ipc",
            "isolated_multi_agent": False,
            "host": host,
            "port": port,
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "rtt_ms": rtt_ms,
            "ack_received": bool(b"ACK" in ack),
        }
        return True, received_payload, meta
    except Exception as e:
        meta = {
            "transport": "tcp_socket",
            "scope": "same_process_loopback_ipc",
            "isolated_multi_agent": False,
            "host": host,
            "port": port,
            "error": str(e),
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received,
            "rtt_ms": None,
            "ack_received": False,
        }
        return False, None, meta
    finally:
        server.close()
