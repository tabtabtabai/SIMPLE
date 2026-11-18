# Copyright (c) 2026 tabtabtab
# SPDX-License-Identifier: MIT
"""Decoupled-WBC agent driven by an external openpi websocket policy (Clockwork 5.4).

Wire action: one [H x 36] float32 chunk per request, in the SIMPLE-decoupled-wbc /
Psi0 layout (USC-PSI-Lab/SIMPLE-decoupled-wbc-Teleop meta/info.json "action"):

   0-6   left hand   thumb_0..2, middle_0..1, index_0..1
   7-13  right hand  thumb_0..2, index_0..1, middle_0..1
  14-20  left arm    shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
  21-27  right arm   same order
  28-30  waist       roll, pitch, yaw
  31     base_height_command (m)
  32-35  navigate    vx, vy, vyaw_or_turning_flag, target_yaw

The legs are never commanded directly: SIMPLE's RL lower-body policy (the decoupled
WBC) tracks these targets and keeps the robot balanced. An arms-only client sends
32-35 = 0 and pins 28-31 to the values in `obs["hold_command"]`.

The WBC is driven on world time (step count x 20 ms), like SIMPLE's replay agent,
not on the host clock as upstream's Psi0 agent does, so a held step in real-time
mode and a slow host both see exactly the dynamics of real time.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from simple.agents.sonic_decoupled_wbc_agent import SonicDecoupledWbcAgent
from simple.core.action import ActionCmd

ACTION_DIM = 36


def upper_body_from_wire(a: np.ndarray) -> np.ndarray:
    """36-d wire action -> decoupled-WBC upper_body joint-group order.

    Same mapping as simple.agents.replay_decoupled_agent._build_wbc_goal (36-d branch).
    """
    return np.concatenate([
        a[30:31], a[28:30],          # waist yaw, roll, pitch
        a[14:21],                    # left arm
        a[5:7], a[3:5], a[0:3],      # left hand index, middle, thumb
        a[21:28],                    # right arm
        a[10:12], a[12:14], a[7:10],  # right hand index, middle, thumb
    ]).astype(np.float32)


def wire_from_state(joint_qpos: np.ndarray, joint_names: list[str], base_height_cmd: float) -> np.ndarray:
    """Current measured pose as a 36-d wire action with zero navigation: the hold command."""
    idx = {n: i for i, n in enumerate(joint_names)}
    order = [
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        *[f"left_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
        *[f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")],
        "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
    ]
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[:31] = [joint_qpos[idx[n]] for n in order]
    out[31] = base_height_cmd
    return out


class ClockworkWsAgent(SonicDecoupledWbcAgent):
    """Queues wire actions; every control step turns the head of the queue (or the
    held command when the queue is empty) into a decoupled-WBC joint command."""

    def __init__(self, robot, sonic_config: dict, **kwargs):
        super().__init__(robot, sonic_config=sonic_config)
        self._queue: deque[np.ndarray] = deque()
        self.last_wire: np.ndarray | None = None
        self.world_t = 0.0

    # --- queue -----------------------------------------------------------
    def enqueue_chunk(self, actions: np.ndarray, skip: int = 0) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"expected [H x {ACTION_DIM}] actions, got {actions.shape}")
        self._queue.clear()
        for row in actions[skip:]:
            self._queue.append(row)

    @property
    def queue_len(self) -> int:
        return len(self._queue)

    def needs_render_next(self) -> bool:
        return len(self._queue) == 0

    # --- per step ----------------------------------------------------------
    def step_command(self, held: bool = False) -> ActionCmd:
        """Joint command for this control step. held=True re-applies the last command."""
        if not held and self._queue:
            self.last_wire = self._queue.popleft()
        if self.last_wire is None:
            raise RuntimeError("no command yet: call engage() first")
        a = self.last_wire
        t_now = self.world_t
        self.world_t += self._control_dt

        proprio = self.robot.prepare_obs()
        self._wbc_policy.set_observation(self._build_wbc_observation(proprio))
        self._wbc_policy.set_goal({
            "target_upper_body_pose": upper_body_from_wire(a),
            "navigate_cmd": a[32:36].astype(np.float64),
            "base_height_command": a[31:32].astype(np.float64),
            "target_time": t_now + 1 / self._control_frequency,
            "interpolation_garbage_collection_time": t_now - 2 / self._control_frequency,
            "timestamp": t_now,
        })
        wbc_action = self._wbc_policy.get_action(time=t_now)
        rm = self._dwbc_robot_model
        self._cached_target_q = rm.get_body_actuated_joints(wbc_action["q"])
        self._cached_left_hand_q = rm.get_hand_actuated_joints(wbc_action["q"], side="left")
        self._cached_right_hand_q = rm.get_hand_actuated_joints(wbc_action["q"], side="right")
        return ActionCmd("decoupled_wbc", target_q=self._cached_target_q,
                         left_hand_q=self._cached_left_hand_q, right_hand_q=self._cached_right_hand_q)

    def engage(self, joint_qpos: np.ndarray, base_height_cmd: float = 0.74) -> np.ndarray:
        """Start the policy phase: the hold command is the measured pose, no navigation."""
        import time

        self._queue.clear()
        self.last_wire = wire_from_state(joint_qpos, list(self.robot.joint_names), base_height_cmd)
        # Stabilization ran the WBC on host-clock timestamps; continue from there so the
        # interpolator never sees time go backwards, then advance by world time only.
        self.world_t = time.monotonic()
        return self.last_wire.copy()

    def reset(self, **kwargs):
        self._queue = deque()
        self.last_wire = None
        self.world_t = 0.0
        return super().reset(**kwargs)
