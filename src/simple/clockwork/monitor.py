# Copyright (c) 2026 tabtabtab
# SPDX-License-Identifier: MIT
"""Read-only episode monitor for Clockwork skill 5.4 (OpenFaucet).

It never calls the task's compute_reward/check_success: those mutate the task's
reward accumulator, so calling them from a wrapper would change the benchmark's
own verdict. The upstream verdict is read from env._success after each env.step.

Everything here is a pure function of per-step samples, so it is unit-tested
without a simulator (tests/clockwork/test_monitor.py).

Definitions (Clockwork CONTRACT.md 3, 3b, D1):
  open            |articulate_joint_0| > OPEN_RAD (0.7 rad), the task's own threshold
  success         upstream: env._success (4 consecutive control steps open, because
                  env.step evaluates compute_reward twice per step, +0.03 each)
  success_strict  open for >= STRICT_HOLD_S continuously, and still open at episode end
  retry           a setback followed by a redo:
                  - setback "fell_back": the handle returns inside +-OPEN_RAD after having
                    been open, before upstream success; the retry is counted when it is
                    opened again
                  - "regrasp": hand-faucet contact lost for >= CONTACT_LOSS_S after having
                    been in contact for >= CONTACT_MIN_S, then regained, before success
  regrasps        count of "regrasp" events (also counted in attempts)
  irreversible    robot_fell: pelvis height < FALL_Z_M or base tilt > FALL_TILT_RAD,
                  sustained for FALL_HOLD_S
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

OPEN_RAD = 0.7
STRICT_HOLD_S = 1.0
CONTACT_MIN_S = 0.10
CONTACT_LOSS_S = 0.25
FALL_Z_M = 0.40
FALL_TILT_RAD = np.deg2rad(60.0)
FALL_HOLD_S = 0.20


def tilt_from_quat_wxyz(q: np.ndarray) -> float:
    """Angle between the base's z axis and world z, from a wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    zz = 1.0 - 2.0 * (x * x + y * y)  # R[2,2]
    return float(np.arccos(np.clip(zz, -1.0, 1.0)))


@dataclass
class FaucetMonitor:
    dt: float  # world seconds per control step
    t: float = 0.0
    # faucet
    q0: float = 0.0
    max_abs_q0: float = 0.0
    was_open: bool = False
    open_since: float | None = None
    longest_open_s: float = 0.0
    fell_back_pending: bool = False
    # upstream verdict
    success: bool = False
    t_success: float | None = None
    # contact
    contact: bool = False
    contact_since: float | None = None
    lost_since: float | None = None
    had_contact: bool = False
    ever_contact: bool = False
    # fall
    low_since: float | None = None
    fell: bool = False
    # events
    retry_events: list[dict[str, Any]] = field(default_factory=list)
    irreversible: list[dict[str, Any]] = field(default_factory=list)
    regrasps: int = 0

    def update(self, *, q0: float, upstream_success: bool, hand_contact: bool,
               pelvis_z: float, tilt_rad: float) -> None:
        """Feed one control step's state, sampled after env.step."""
        self.t += self.dt
        t = self.t
        self.q0 = float(q0)
        self.max_abs_q0 = max(self.max_abs_q0, abs(self.q0))
        is_open = abs(self.q0) > OPEN_RAD

        # --- open / fell back / strict hold ---
        if is_open:
            if self.open_since is None:
                self.open_since = t
                if self.fell_back_pending:
                    self.retry_events.append({"t": round(t, 3), "kind": "reopen_after_fall_back"})
                    self.fell_back_pending = False
            self.longest_open_s = max(self.longest_open_s, t - self.open_since + self.dt)
        else:
            if self.open_since is not None and not self.success:
                self.fell_back_pending = True
            self.open_since = None
        self.was_open = is_open

        if upstream_success and not self.success:
            self.success = True
            self.t_success = t

        # --- hand-faucet contact, debounced ---
        if hand_contact:
            self.ever_contact = True
            if self.contact_since is None:
                self.contact_since = t
            if self.lost_since is not None:
                lost_for = t - self.lost_since
                if self.had_contact and lost_for >= CONTACT_LOSS_S and not self.success:
                    self.regrasps += 1
                    self.retry_events.append({"t": round(t, 3), "kind": "regrasp",
                                              "lost_s": round(lost_for, 3)})
                self.lost_since = None
            if t - self.contact_since + self.dt >= CONTACT_MIN_S:
                self.had_contact = True
        else:
            if self.contact_since is not None and self.lost_since is None:
                self.lost_since = t
            self.contact_since = None
        self.contact = hand_contact

        # --- fall ---
        low = pelvis_z < FALL_Z_M or tilt_rad > FALL_TILT_RAD
        if low:
            if self.low_since is None:
                self.low_since = t
            if not self.fell and t - self.low_since + self.dt >= FALL_HOLD_S:
                self.fell = True
                self.irreversible.append({"t": round(self.low_since, 3), "kind": "robot_fell",
                                          "pelvis_z": round(float(pelvis_z), 3),
                                          "tilt_deg": round(float(np.rad2deg(tilt_rad)), 1)})
        else:
            self.low_since = None

    @property
    def is_open(self) -> bool:
        return abs(self.q0) > OPEN_RAD

    @property
    def strict_met(self) -> bool:
        """Open continuously for STRICT_HOLD_S right now."""
        return self.open_since is not None and (self.t - self.open_since + self.dt) >= STRICT_HOLD_S - 1e-9

    def success_strict(self) -> bool:
        return self.strict_met

    @property
    def attempts(self) -> int:
        return 1 + len(self.retry_events)

    def detail(self) -> str:
        sign = "+" if self.q0 >= 0 else "-"
        return (f"final articulate_joint_0={self.q0:+.3f} rad ({sign}{np.rad2deg(abs(self.q0)):.0f} deg), "
                f"max |q0|={self.max_abs_q0:.3f} rad, longest open {self.longest_open_s:.2f} s, "
                f"upstream success at t={self.t_success if self.t_success is not None else 'never'}, "
                f"hand-faucet contact {'seen' if self.ever_contact else 'never'}")

    def progress(self) -> float:
        """Partial credit, reported only: fraction of the 0.7 rad opening reached."""
        return float(min(1.0, self.max_abs_q0 / OPEN_RAD))
