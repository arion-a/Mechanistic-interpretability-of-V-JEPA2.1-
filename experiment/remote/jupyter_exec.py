"""Thin remote-execution client over a Jupyter kernel's WebSocket channel.

Lets us drive the RunPod GPU pod's Jupyter server directly over HTTPS/WSS
(no SSH needed - this sandbox's network policy blocks raw TCP but allows
this). One kernel is kept alive across calls so state (loaded models,
installed packages, current directory) persists between exec() calls.

Not committed to the repo (credentials): keep pod URL/token out of git.
"""
from __future__ import annotations

import json
import time
import uuid

import requests
import websocket


class RemoteKernel:
    def __init__(self, base_url: str, token: str, kernel_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.ws_base = self.base_url.replace("https://", "wss://").replace("http://", "ws://")
        self.token = token
        self.kernel_id = kernel_id or self._start_kernel()
        self.ws = None
        self._connect()

    def _start_kernel(self) -> str:
        r = requests.post(f"{self.base_url}/api/kernels", params={"token": self.token},
                           json={"name": "python3"}, timeout=30)
        r.raise_for_status()
        return r.json()["id"]

    def _connect(self):
        url = f"{self.ws_base}/api/kernels/{self.kernel_id}/channels?token={self.token}"
        self.ws = websocket.create_connection(url, timeout=30)

    def exec(self, code: str, timeout: float = 120.0) -> dict:
        """Execute code in the persistent kernel. Returns {stdout, stderr, error, result}."""
        msg_id = str(uuid.uuid4())
        msg = {
            "header": {"msg_id": msg_id, "msg_type": "execute_request", "username": "claude",
                       "session": str(uuid.uuid4()), "version": "5.3"},
            "parent_header": {}, "metadata": {},
            "content": {"code": code, "silent": False, "store_history": True},
            "channel": "shell",
        }
        self.ws.send(json.dumps(msg))

        stdout_parts, stderr_parts, result_parts = [], [], []
        error = None
        t0 = time.time()
        self.ws.settimeout(5.0)
        while time.time() - t0 < timeout:
            try:
                raw = self.ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                error = f"websocket error: {e}"
                break
            m = json.loads(raw)
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = m["msg_type"]
            content = m.get("content", {})
            if mtype == "stream":
                (stdout_parts if content.get("name") == "stdout" else stderr_parts).append(content.get("text", ""))
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "text/plain" in data:
                    result_parts.append(data["text/plain"])
            elif mtype == "error":
                error = "\n".join(content.get("traceback", [content.get("evalue", "unknown error")]))
            elif mtype == "execute_reply":
                break
        return {
            "stdout": "".join(stdout_parts),
            "stderr": "".join(stderr_parts),
            "result": "".join(result_parts),
            "error": error,
            "timed_out": (time.time() - t0) >= timeout and error is None,
        }

    def close(self):
        if self.ws:
            self.ws.close()
        try:
            requests.delete(f"{self.base_url}/api/kernels/{self.kernel_id}", params={"token": self.token}, timeout=10)
        except Exception:
            pass


def upload_file(base_url: str, token: str, remote_path: str, local_path: str, is_text: bool = True):
    """Write a local file's contents to the pod's filesystem via the Contents API."""
    import base64
    with open(local_path, "rb") as f:
        data = f.read()
    if is_text:
        payload = {"type": "file", "format": "text", "content": data.decode("utf-8")}
    else:
        payload = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode("ascii")}
    r = requests.put(f"{base_url.rstrip('/')}/api/contents/{remote_path}",
                      params={"token": token}, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()
