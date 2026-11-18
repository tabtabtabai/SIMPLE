# Copyright (c) 2026 tabtabtab
# SPDX-License-Identifier: MIT
"""openpi websocket policy servers for Clockwork 5.4.

  replay  the reference "expert": the recorded human teleop actions of one demo,
          served in chunks on the demo's own timeline. Pair it with
          `run --data-dir <teleop set> --episode k` so the env resets to that demo's layout.
          After the recording ends it holds the last pose with zero navigation.
  psi0    a bridge to an upstream Psi0 HTTP server (Psi0 repo serve_psi0_simple.sh): builds
          exactly the request simple.baselines.psi0_decoupled_wbc sends (head_stereo_left,
          32-d state, instruction, history reset) and returns its [H x 36] chunk unchanged.

    python -m simple.clockwork.servers replay --data-dir data/SIMPLE-decoupled-wbc-Teleop --port 8000
    python -m simple.clockwork.servers psi0 --psi0-host 10.x.x.x --psi0-port 22085 --port 8000
"""

from __future__ import annotations

from typing import Annotated

import numpy as np
import typer

from simple.clockwork.wire import serve

app = typer.Typer(add_completion=False)

# simple.baselines.psi0_decoupled_wbc.STATE_SLICES, in the robot's joint_qpos order
PSI0_STATE_SLICES = [(29, 32), (34, 36), (32, 34), (36, 43), (15, 22), (22, 29)]


@app.command()
def replay(
    data_dir: Annotated[str, typer.Option()],
    port: Annotated[int, typer.Option()] = 8000,
    chunk: Annotated[int, typer.Option(help="rows per reply (50 = 1 s)")] = 50,
):
    import pandas as pd

    from simple.clockwork.run import task_episode_ids
    from simple.datasets.lerobot import LeRobotParquetDataset

    ids = task_episode_ids(data_dir)
    ds = LeRobotParquetDataset(data_dir)
    state = {"rows": None, "cursor": 0, "episode": None}
    print(f"[replay] {len(ids)} OpenFaucet demos in {data_dir}", flush=True)

    def load(k: int):
        g = ids[k]
        lo = int(ds.episode_data_index["from"][g])
        hi = int(ds.episode_data_index["to"][g])
        acts = np.stack([np.asarray(ds.hf_dataset[i]["action"], dtype=np.float32) for i in range(lo, hi)])
        state.update(rows=acts, cursor=0, episode=k)
        print(f"[replay] demo #{k} (global {g}): {len(acts)} rows = {len(acts) / 50:.2f} s", flush=True)

    def policy(obs: dict) -> dict:
        k = int(obs.get("episode", obs.get("seed", 0)))
        if obs.get("reset") or state["episode"] != k:
            load(k)
        rows, c = state["rows"], state["cursor"]
        out = rows[c:c + chunk]
        if len(out) < chunk:  # recording over: hold the last pose, stop walking
            pad = np.repeat(rows[-1:], chunk - len(out), axis=0)
            pad[:, 32:35] = 0.0
            out = np.concatenate([out, pad]) if len(out) else pad
        state["cursor"] = c + chunk
        return {"actions": out.astype(np.float32), "demo_rows": int(len(rows)), "cursor": int(c)}

    serve(policy, {"policy": "teleop-replay", "action_space": "simple-decoupled-wbc-36", "chunk": chunk}, port=port)


@app.command()
def psi0(
    psi0_host: Annotated[str, typer.Option()],
    psi0_port: Annotated[int, typer.Option()] = 22085,
    port: Annotated[int, typer.Option()] = 8000,
):
    from simple.baselines.client import HttpActionClient

    http = HttpActionClient(psi0_host, psi0_port)

    def policy(obs: dict) -> dict:
        q = np.asarray(obs["state"], dtype=np.float32)[None]
        waist_rpy = q[:, [13, 14, 12]]
        height = np.array([[float(np.asarray(obs["hold_command"])[31])]], dtype=np.float32)
        states = np.concatenate([q[:, s:e] for s, e in PSI0_STATE_SLICES] + [waist_rpy, height], axis=1)
        history = {"reset": True} if obs.get("reset") else {}
        action, *_ = http.query_action(
            {"rgb_head_stereo_left": obs["images"]["head_stereo_left"]},
            obs["prompt"], {"states": states.astype(np.float32)}, {},
            history=history, dataset="simple",
        )
        return {"actions": np.asarray(action, dtype=np.float32)}

    serve(policy, {"policy": "psi0-bridge", "upstream": f"http://{psi0_host}:{psi0_port}",
                   "action_space": "simple-decoupled-wbc-36"}, port=port)


if __name__ == "__main__":
    app()
