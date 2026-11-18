"""Unit tests for the Clockwork 5.4 episode monitor and wire format (no simulator)."""
import numpy as np
import pytest

from simple.clockwork.monitor import FaucetMonitor, OPEN_RAD, tilt_from_quat_wxyz

DT = 0.02
UP = dict(pelvis_z=0.74, tilt_rad=0.0)


def feed(m, qs, success_from=None, contact=True):
    for i, q in enumerate(qs):
        m.update(q0=q, upstream_success=success_from is not None and i >= success_from,
                 hand_contact=contact if not callable(contact) else contact(i), **UP)


def test_clean_open_is_one_attempt_and_strict():
    m = FaucetMonitor(dt=DT)
    feed(m, [0.0] * 10 + [0.8] * 60, success_from=13)
    assert m.success and m.attempts == 1 and m.regrasps == 0
    assert m.strict_met and m.success_strict()
    assert m.t_success == pytest.approx(14 * DT)


def test_negative_direction_counts():
    m = FaucetMonitor(dt=DT)
    feed(m, [-0.9] * 55, success_from=3)
    assert m.success and m.strict_met and m.q0 < -OPEN_RAD


def test_fall_back_then_reopen_is_a_retry():
    m = FaucetMonitor(dt=DT)
    feed(m, [0.75, 0.75, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8], success_from=7)
    assert [e["kind"] for e in m.retry_events] == ["reopen_after_fall_back"]
    assert m.attempts == 2


def test_fall_back_after_success_is_not_a_retry_but_fails_strict():
    m = FaucetMonitor(dt=DT)
    feed(m, [0.8] * 5 + [0.3] * 5, success_from=3)
    assert m.success and m.attempts == 1 and not m.strict_met


def test_contact_flicker_is_not_a_regrasp():
    m = FaucetMonitor(dt=DT)
    # 10 steps contact, 5 steps (0.1 s < 0.25 s) lost, contact again
    feed(m, [0.1] * 25, contact=lambda i: not (10 <= i < 15))
    assert m.regrasps == 0 and m.attempts == 1


def test_slip_and_regrasp_counts():
    m = FaucetMonitor(dt=DT)
    # handle turned 0 -> 0.3 while gripped, grip lost 0.3 s, taken again
    qs = [0.0] * 5 + [0.3] * 35
    feed(m, qs, contact=lambda i: not (10 <= i < 25))
    assert m.regrasps == 1 and m.attempts == 2


def test_approach_brush_then_grasp_is_not_a_regrasp():
    """Replay seed 0: the hand brushes the faucet, loses it for 1.4 s, then grasps and turns."""
    m = FaucetMonitor(dt=DT)
    qs = [0.0] * 200 + [i * 0.01 for i in range(100)]
    feed(m, qs, contact=lambda i: (100 <= i < 110) or i >= 180)
    assert m.regrasps == 0 and m.attempts == 1


def test_first_approach_contact_is_not_a_regrasp():
    m = FaucetMonitor(dt=DT)
    feed(m, [0.0] * 40, contact=lambda i: i >= 30)
    assert m.regrasps == 0


def test_robot_fell_needs_sustained_low_pelvis():
    m = FaucetMonitor(dt=DT)
    for _ in range(5):
        m.update(q0=0, upstream_success=False, hand_contact=False, pelvis_z=0.3, tilt_rad=0)
    assert not m.fell  # 0.1 s < 0.2 s
    for _ in range(10):
        m.update(q0=0, upstream_success=False, hand_contact=False, pelvis_z=0.3, tilt_rad=0)
    assert m.fell and m.irreversible[0]["kind"] == "robot_fell"


def test_tilt_from_quat():
    assert tilt_from_quat_wxyz(np.array([1, 0, 0, 0])) == pytest.approx(0)
    s = np.sin(np.pi / 4)  # 90 deg about x
    assert tilt_from_quat_wxyz(np.array([s, s, 0, 0])) == pytest.approx(np.pi / 2)


def test_wire_roundtrip():
    from simple.clockwork.wire import packb, unpackb
    obs = {"images": {"a": np.arange(12, dtype=np.uint8).reshape(2, 2, 3)}, "t": 1.5,
           "state": np.ones(43, np.float32), "prompt": "turn the faucet"}
    back = unpackb(packb(obs))
    assert back["prompt"] == "turn the faucet"
    np.testing.assert_array_equal(back["images"]["a"], obs["images"]["a"])
    assert back["state"].dtype == np.float32


def test_physics_qa_flags():
    from simple.clockwork.monitor import PhysicsQA
    qa = PhysicsQA()
    for q in (0.0, 0.0, 0.02):  # 1.1 deg settle drift
        qa.settle(q, [0, 0, 0.7])
    qa.step(pelvis_z=0.74, tilt_rad=0.01, contact_force_n=3.0, penetration_m=0.001, joint_limit_violation=False)
    qa.step(pelvis_z=0.70, tilt_rad=0.02, contact_force_n=5.0, penetration_m=0.0025, joint_limit_violation=True)
    r = qa.report()
    assert r["max_contact_force_n"] == 5.0 and r["joint_limit_violations"] == 1
    assert r["base_height_range_m"] == 0.04
    assert set(r["flags"]) == {"settle_drift_deg", "max_penetration_mm", "joint_limit_violations", "base_height_range_m"}
