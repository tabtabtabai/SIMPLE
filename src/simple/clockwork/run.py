# Copyright (c) 2026 tabtabtab
# SPDX-License-Identifier: MIT
"""Clockwork 5.4 episode runner: SIMPLE OpenFaucet behind an openpi websocket policy.

    python -m simple.clockwork.run --data-dir data/SIMPLE-eval/G1WholebodyOpenFaucetTeleop-v0/dr-level-0 \
        --episode 0 --host POLICY_HOST --port 8000 --mode realtime --policy-id psi0-openfaucet-40k \
        --out-dir data/clockwork/5.4

One episode per process (Isaac does not tear down cleanly). Writes
<out-dir>/<policy-id>/<mode>/<split>_ep<k>.json and, with --video, an mp4 next to it.

Clock (CONTRACT.md 2, 3a): world time starts when the policy is engaged, after SIMPLE's
own stabilization, and advances 20 ms per control step. In real-time mode a request's
client-timed latency is paid in world time with the last command held: control steps
run while the request is out (never ahead of the wall time since it was sent), and any
remainder is run before the reply's first action. A chunk that plays back without a new
request costs only its own control steps. In lockstep mode the world waits.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

ENV_ID = "simple/G1WholebodyOpenFaucetTeleop-v0"
CONTROL_DT = 0.02
NATIVE_MAX_STEPS = 1000  # task metadata max_episode_steps (20 s)

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------- sim probes
class MjProbe:
    """Read-only MuJoCo probes: faucet angle, hand-faucet contact, pelvis pose, cameras."""

    def __init__(self, mj_engine):
        import mujoco

        self.mujoco = mujoco
        self.eng = mj_engine
        m = mj_engine.mjModel
        self.m = m
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "articulate_base")
        self.faucet_bodies = set()
        for b in range(m.nbody):
            p = b
            while p > 0 and p != base:
                p = m.body_parentid[p]
            if p == base and base >= 0:
                self.faucet_bodies.add(b)
        self.hand_bodies = set()
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if name.startswith(("left_", "right_")) and ("hand" in name or "wrist" in name):
                self.hand_bodies.add(b)
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if not self.faucet_bodies or not self.hand_bodies or self.pelvis < 0:
            raise RuntimeError(f"probe setup failed: faucet={len(self.faucet_bodies)} "
                               f"hands={len(self.hand_bodies)} pelvis={self.pelvis}")

    @property
    def d(self):
        return self.eng.mjData

    def q0(self) -> float:
        return float(self.d.joint("articulate_joint_0").qpos[0])

    def hand_contact(self) -> bool:
        gb = self.m.geom_bodyid
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = gb[c.geom1], gb[c.geom2]
            if (b1 in self.hand_bodies and b2 in self.faucet_bodies) or \
               (b2 in self.hand_bodies and b1 in self.faucet_bodies):
                return True
        return False

    def pelvis_T_world(self) -> np.ndarray:
        """4x4 world<-pelvis transform."""
        T = np.eye(4, dtype=np.float32)
        q = np.zeros(9)
        self.mujoco.mju_quat2Mat(q, self.d.xquat[self.pelvis])
        T[:3, :3] = q.reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.pelvis]
        return T

    def pelvis_pose(self) -> tuple[float, np.ndarray]:
        return float(self.d.xpos[self.pelvis][2]), np.array(self.d.xquat[self.pelvis])

    def camera_info(self, names: list[str], hw: dict[str, tuple[int, int]]) -> dict[str, Any]:
        """K (3x3) and T_world_cam (4x4, OpenCV axes: x right, y down, z forward) per camera."""
        out = {}
        flip = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera looks down -z with y up
        for n in names:
            cid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_CAMERA, n)
            if cid < 0 or n not in hw:
                continue
            h, w = hw[n]
            fy = 0.5 * h / math.tan(math.radians(self.m.cam_fovy[cid]) / 2)
            K = np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = self.d.cam_xmat[cid].reshape(3, 3) @ flip
            T[:3, 3] = self.d.cam_xpos[cid]
            out[n] = {"K": K, "T_world_cam": T, "source": "mujoco"}
        return out


# ---------------------------------------------------------------------------- dataset
TASK_UID = "g1_wholebody_open_faucet_teleop"


def task_episode_ids(data_dir: str, task_uid: str = TASK_UID) -> list[int]:
    """Positions (as LeRobotParquetDataset / get_episode_lerobot index them: episodes sorted by
    episode_index) of this task's episodes.

    The eval set holds one task per dir; the merged teleop set holds 12 tasks, so seed k
    means the k-th OpenFaucet episode there."""
    eps = []
    with open(Path(data_dir) / "meta" / "episodes.jsonl") as f:
        for line in f:
            if line.strip():
                eps.append(json.loads(line))
    eps.sort(key=lambda e: e["episode_index"])
    out = []
    for pos, ep in enumerate(eps):
        cfg = ep.get("environment_config")
        cfg = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
        if cfg.get("uid") == task_uid:
            out.append(pos)
    return out


def load_episode(data_dir: str, k: int) -> tuple[dict, int]:
    from simple.datasets.lerobot import LeRobotParquetDataset, get_episode_lerobot

    ids = task_episode_ids(data_dir)
    if k >= len(ids):
        raise IndexError(f"{data_dir} has {len(ids)} {TASK_UID} episodes, asked for #{k}")
    ds = LeRobotParquetDataset(data_dir)
    env_conf, _ = get_episode_lerobot(ds, ids[k], load_videos=False)
    return env_conf, int(ds.meta.episodes[ids[k]]["episode_index"])


# ---------------------------------------------------------------------------- main
@app.command()
def main(
    data_dir: Annotated[str, typer.Option(help="LeRobot dir with environment_config per episode")],
    episode: Annotated[int, typer.Option()] = 0,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    mode: Annotated[str, typer.Option(help="realtime | charged | lockstep (CONTRACT 2, D9)")] = "realtime",
    policy_id: Annotated[str, typer.Option()] = "unknown",
    split: Annotated[str, typer.Option(help="label for the record, e.g. dr-level-0 or teleop")] = "",
    budget_s: Annotated[float, typer.Option(help="world-time limit; contract: max(60, 6*ref_s)")] = 60.0,
    strict_tail: Annotated[bool, typer.Option(help="after upstream success keep running until success_strict is decided")] = True,
    out_dir: Annotated[str, typer.Option()] = "data/clockwork/5.4",
    video: Annotated[bool, typer.Option("--video/--no-video")] = True,
    video_held_steps: Annotated[bool, typer.Option(help="render held steps too, so the video plays world time at 1x")] = True,
    video_cameras: Annotated[str, typer.Option(help="one mp4 per camera, every control step (D8)")] = "head_stereo_left,head_stereo_right,third_person",
    gcs_prefix: Annotated[str, typer.Option(help="where the pipeline uploads this episode's files; recorded in `video`")] = "",
    sim_mode: Annotated[str, typer.Option()] = "mujoco_isaac",
    send_cameras: Annotated[str, typer.Option(help="comma list of image keys to send")] = "head_stereo_left,head_stereo_right,third_person",
    pace: Annotated[bool, typer.Option(help="never let world time run ahead of wall time (as upstream)")] = True,
    variant: Annotated[str, typer.Option(help="native | near-start (scored separately)")] = "native",
    free_period: Annotated[bool, typer.Option("--free-period/--no-free-period",
        help="program-step policy: a request owes only latency beyond the previous chunk's control periods")] = False,
):
    import gymnasium as gym

    import simple.envs as _  # noqa: F401
    from simple.cli.eval_decoupled_wbc import _make_sonic_config
    from simple.clockwork.agent import ClockworkWsAgent
    from simple.clockwork.monitor import FaucetMonitor, tilt_from_quat_wxyz
    from simple.clockwork.wire import WsPolicyClient

    assert mode in ("realtime", "charged", "lockstep"), mode
    t_host0 = time.monotonic()
    split = split or Path(data_dir).name
    out = Path(out_dir) / policy_id / mode
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_ep{episode}"

    env_conf, global_episode = load_episode(data_dir, episode)
    sonic_config = _make_sonic_config()
    raw_env = gym.make(ENV_ID, sim_mode=sim_mode, render_hz=50, headless=True, sonic_config=sonic_config)
    senv = raw_env.unwrapped
    task = senv.task
    on_query = senv.isaac is not None
    if on_query:
        senv.sync_isaac_every_substep = False
    robot = task.robot
    agent = ClockworkWsAgent(robot, sonic_config=sonic_config)
    client = WsPolicyClient(host, port)
    pool = cf.ThreadPoolExecutor(max_workers=1)

    observation, info = senv.reset(options={"state_dict": env_conf})
    agent.reset()
    agent._wbc_policy.lower_body_policy.use_policy_action = True
    probe = MjProbe(senv.mujoco)
    print(f"[clockwork] task success_criteria={task.metadata.get('success_criteria')} "
          f"faucet bodies={len(probe.faucet_bodies)} hand bodies={len(probe.hand_bodies)}", flush=True)

    # --- SIMPLE's own stabilization (not on the clock) ---
    senv.render_isaac = not on_query
    n = 0
    while not robot.stabilized and n < 300:
        observation, *_ = senv.step(agent.get_stabilize_action(observation))
        n += 1
    import torch
    agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1), dtype=torch.float32)

    writers: dict[str, Any] = {}
    vid_keys = [k for k in video_cameras.split(",") if k]

    def render_obs():
        senv.render_isaac = True
        return senv.refresh_obs()

    observation = render_obs()
    hold = agent.engage(observation["joint_qpos"])
    mon = FaucetMonitor(dt=CONTROL_DT)
    prev_chunk_len = 0
    charged_world_s = 0.0  # charged mode: latency added to world time without stepping
    stats = dict(requests=0, latency_s=0.0, charged_s=0.0, held_steps=0, acting_steps=0, paced_s=0.0, max_latency_s=0.0)
    instruction = task.instruction
    world_steps = 0
    budget_steps = int(round(budget_s / CONTROL_DT))
    t_native_success = None
    wall_anchor = time.monotonic()  # pacing anchor: wall time when world_steps was 0

    def cams() -> list[str]:
        return [c for c in send_cameras.split(",") if c in observation]

    def step_once(held: bool, render: bool):
        nonlocal observation, info, world_steps
        senv.render_isaac = render or not on_query
        cmd = agent.step_command(held=held)
        observation, _r, _term, _trunc, info = senv.step(cmd)
        world_steps += 1
        stats["held_steps" if held else "acting_steps"] += 1
        z, quat = probe.pelvis_pose()
        mon.update(q0=probe.q0(), upstream_success=bool(senv._success), hand_contact=probe.hand_contact(),
                   pelvis_z=z, tilt_rad=tilt_from_quat_wxyz(quat))
        if video and render:
            for k in vid_keys:
                if k not in observation:
                    continue
                if k not in writers:
                    import imageio.v2 as imageio
                    writers[k] = imageio.get_writer(out / f"{stem}_{k}.mp4", fps=int(round(1 / CONTROL_DT)),
                                                    macro_block_size=1, quality=7)
                writers[k].append_data(np.asarray(observation[k]))
        if pace:
            ahead = world_steps * CONTROL_DT - (time.monotonic() - wall_anchor)
            if ahead > 0:
                time.sleep(ahead)
                stats["paced_s"] += ahead

    def build_request() -> dict:
        names = cams()
        hw = {k: observation[k].shape[:2] for k in names}
        return {
            "images": {k: np.asarray(observation[k], dtype=np.uint8) for k in names},
            "camera_info": probe.camera_info(names, hw),
            "pelvis_pose_world": probe.pelvis_T_world(),
            "state": np.asarray(observation["joint_qpos"], dtype=np.float32),
            "state_names": list(robot.joint_names),
            "hold_command": agent.last_wire.copy(),
            "prompt": instruction,
            "t_robot": world_steps * CONTROL_DT + charged_world_s,
            "reset": stats["requests"] == 0,
            "episode": episode,
            "robot": "unitree_g1_dex3_decoupled_wbc",
            "action_space": "simple-decoupled-wbc-36",
        }

    def done() -> bool:
        if mon.fell or world_steps * CONTROL_DT + charged_world_s >= budget_s - 1e-9:
            return True
        if mon.success and (not strict_tail or mon.strict_met):
            return True
        return False

    # ------------------------------------------------------------------ policy loop
    while not done():
        if agent.queue_len == 0:
            req = build_request()
            stats["requests"] += 1
            fut = pool.submit(client.infer, req)
            if mode == "realtime":
                t_sent = time.monotonic()
                held = 0
                free = prev_chunk_len * CONTROL_DT if free_period else 0.0
                while not fut.done() and not done():
                    if held * CONTROL_DT < time.monotonic() - t_sent - free:  # never ahead of wall time
                        step_once(held=True, render=video_held_steps and video)
                        held += 1
                    else:
                        time.sleep(0.001)
                reply, latency = fut.result()
                # --free-period (a program-step policy, as RoboLab real-time mode): a request owes
                # only the latency beyond the control periods the previous chunk covered, since on
                # a robot it computes while that chunk moves. Otherwise it owes its whole latency.
                charged = max(0.0, latency - free)
                stats["charged_s"] += charged
                owed = int(math.ceil(max(0.0, charged - held * CONTROL_DT) / CONTROL_DT - 1e-6))
                for _ in range(owed):
                    if done():
                        break
                    step_once(held=True, render=video_held_steps and video)
                    held += 1
            else:
                reply, latency = fut.result()
                wall_anchor += latency  # the physics is frozen while waiting
                if mode == "charged":
                    # D9: the world stands still while the request is out, and its latency is
                    # added to world time and the robot clock (free period as in realtime).
                    c = max(0.0, latency - (prev_chunk_len * CONTROL_DT if free_period else 0.0))
                    charged_world_s += c
                    stats["charged_s"] += c
                    mon.t += c  # monitor clock: a frozen open handle stays open for c
            stats["latency_s"] += latency
            stats["max_latency_s"] = max(stats["max_latency_s"], latency)
            if done():
                break
            agent.enqueue_chunk(np.asarray(reply["actions"]))
            prev_chunk_len = len(reply["actions"])
        step_once(held=False, render=agent.queue_len == 0 or video)
        if mon.success and t_native_success is None:
            t_native_success = mon.t_success

    # ------------------------------------------------------------------ record
    t_world = world_steps * CONTROL_DT + charged_world_s
    success = mon.success and not mon.fell
    success_native = success and mon.t_success is not None and mon.t_success <= NATIVE_MAX_STEPS * CONTROL_DT
    rec = {
        "test_id": "5.4", "benchmark": "SIMPLE", "task": "OpenFaucet", "variant": variant,
        "env_id": ENV_ID, "robot": "Unitree G1 whole-body (Dex3, decoupled WBC)",
        "sim": f"MuJoCo 3.12 physics + Isaac Sim 6.2 render ({sim_mode})",
        "seed": episode, "split": split, "data_dir": data_dir, "dataset_episode_index": global_episode,
        "policy_id": policy_id, "mode": mode,
        "success": bool(success),
        "success_native": bool(success_native),
        "success_strict": bool(mon.success_strict() and not mon.fell),
        "success_detail": mon.detail(),
        "progress": round(mon.progress(), 3),
        "t_wall_s": round(t_world, 3),           # world seconds (3a): sim time incl. charged latency
        "t_world_s": round(t_world, 3),
        "t_success_s": mon.t_success,
        "t_host_s": round(time.monotonic() - t_host0, 1),
        "policy_latency_s": round(stats["latency_s"], 3),
        "charged_latency_s": round(stats["charged_s"], 3),
        "requests": stats["requests"], "max_latency_s": round(stats["max_latency_s"], 3),
        "held_steps": stats["held_steps"], "acting_steps": stats["acting_steps"],
        "t_physics_s": round(world_steps * CONTROL_DT, 3), "charged_world_s": round(charged_world_s, 3),
        "attempts": mon.attempts, "regrasps": mon.regrasps,
        "retry_events": mon.retry_events, "irreversible": mon.irreversible,
        "budget_s": budget_s, "final_q0_rad": round(mon.q0, 4), "free_period": free_period,
        "instruction": instruction, "video": None,
        "server_metadata": client.metadata if isinstance(client.metadata, dict) else None,
    }
    if writers:
        for w in writers.values():
            w.close()
        names = [f"{stem}_{k}.mp4" for k in writers]
        rec["video"] = [f"{gcs_prefix.rstrip('/')}/{n}" if gcs_prefix else str(out / n) for n in names]
    (out / f"{stem}.json").write_text(json.dumps(rec, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: rec[k] for k in ("success", "success_strict", "t_wall_s", "attempts", "irreversible", "success_detail")}), flush=True)
    client.close()
    pool.shutdown(wait=False)
    os._exit(0)  # Isaac's shutdown hangs; the record is already on disk


if __name__ == "__main__":
    app()
