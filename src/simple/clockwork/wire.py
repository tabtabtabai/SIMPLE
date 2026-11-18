# Copyright (c) 2026 tabtabtab
# SPDX-License-Identifier: MIT
"""openpi websocket wire format (msgpack + numpy), compatible with openpi_client.

Same encoding as openpi_client.msgpack_numpy: an ndarray is a map with keys
b"__ndarray__", b"data", b"dtype", b"shape". On connect the server sends one
metadata message; each request is one obs map and each reply one map holding
"actions". Vendored (40 lines) so the SIMPLE image needs only msgpack and websockets.
"""

from __future__ import annotations

import functools
import time
from typing import Any

import msgpack
import numpy as np


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": np.ascontiguousarray(obj).tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: dict) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class WsPolicyClient:
    """Minimal synchronous openpi client. infer() returns (reply, client-timed latency s)."""

    def __init__(self, host: str, port: int, connect_timeout_s: float = 600.0):
        from websockets.sync.client import connect

        uri = f"ws://{host}:{port}"
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self._ws = connect(uri, compression=None, max_size=None, open_timeout=30)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(2.0)
        self.metadata = unpackb(self._ws.recv())

    def infer(self, obs: dict) -> tuple[dict, float]:
        t0 = time.monotonic()
        self._ws.send(packb(obs))
        reply = self._ws.recv()
        latency = time.monotonic() - t0
        if isinstance(reply, str):
            raise RuntimeError(f"policy server error:\n{reply}")
        return unpackb(reply), latency

    def reset(self, seed: int | None = None) -> None:
        # openpi has no reset message; servers here accept {"reset": True} in the next obs.
        pass

    def close(self) -> None:
        self._ws.close()


def serve(policy_fn, metadata: dict, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Blocking openpi-style server: policy_fn(obs dict) -> reply dict with "actions"."""
    import traceback

    from websockets.sync.server import serve as ws_serve

    def handler(ws):
        ws.send(packb(metadata))
        while True:
            try:
                obs = unpackb(ws.recv())
            except Exception:
                return
            try:
                t0 = time.monotonic()
                reply = policy_fn(obs)
                reply.setdefault("server_timing", {"infer_ms": (time.monotonic() - t0) * 1e3})
                ws.send(packb(reply))
            except Exception:
                ws.send(traceback.format_exc())
                return

    with ws_serve(handler, host, port, compression=None, max_size=None) as server:
        print(f"[clockwork] policy server listening on ws://{host}:{port}", flush=True)
        server.serve_forever()
