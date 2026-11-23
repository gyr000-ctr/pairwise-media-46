"""camerashake 边界契约自测：频谱集中、幅度上界、非法输入、性能、并发。"""

import math
import threading
import time

import pytest

import camerashake
from camerashake import CameraShake


def _goertzel(samples, freq, dt):
    """单频功率（Goertzel），无第三方依赖。"""
    w = 2.0 * math.pi * freq * dt
    coeff = 2.0 * math.cos(w)
    s0 = s1 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s0
        s1 = s0
    return s0 * s0 + s1 * s1 - coeff * s0 * s1


def test_shake_band_energy_is_concentrated():
    dt = 0.001
    s = CameraShake(1.0, 4.0, 12345)
    samples = [s.update(dt)[0] for _ in range(1500)]
    mean = sum(samples) / len(samples)
    samples = [x - mean for x in samples]

    freqs = [i * 0.25 for i in range(1, 49)]  # 0.25 .. 12 Hz
    powers = {f: _goertzel(samples, f, dt) for f in freqs}
    total = sum(powers.values())
    lo, hi = CameraShake.BAND_HZ
    in_band = sum(p for f, p in powers.items() if lo <= f <= hi)
    assert in_band / total > 0.85, (in_band / total, total)

    peak = max(powers.values())
    strong = [f for f, p in powers.items() if p >= 0.25 * peak]
    assert len(strong) >= 3, strong  # 多频率分量，而非单一正弦


def test_shake_displacement_never_exceeds_strength():
    strength = 2.5
    s = CameraShake(strength, 3.0, 9)
    worst = 0.0
    for _ in range(6000):
        dx, dy = s.update(1 / 240.0)
        worst = max(worst, abs(dx), abs(dy), math.hypot(dx, dy))
    assert worst <= strength + 1e-12
    assert worst > 0.05 * strength  # 确实有抖动，不是恒零


def test_shake_same_seed_reproducible_and_seed_changes_output():
    def run(seed):
        s = CameraShake(1.0, 2.0, seed)
        return [s.update(1 / 120.0) for _ in range(300)]

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_shake_decays_to_exact_zero_and_stays():
    s = CameraShake(1.0, 1.0, 3)
    prev = s.remaining()
    for _ in range(9):
        s.update(0.1)
        cur = s.remaining()
        assert 0.0 <= cur <= prev
        prev = cur
    s.update(0.1)
    assert s.remaining() == 0.0
    for _ in range(5):
        assert s.update(0.1) == (0.0, 0.0)
        assert s.remaining() == 0.0


def test_shake_zero_dt_is_zero_and_fresh_instance_independent():
    a = CameraShake(1.0, 1.0, 7)
    b = CameraShake(1.0, 1.0, 7)
    assert a.update(0.0) == (0.0, 0.0)
    for _ in range(10):
        a.update(0.1)
    assert a.remaining() == 0.0
    assert b.update(0.0) == (0.0, 0.0)
    assert b.remaining() == 1.0  # b 未推进，不受 a 的状态影响


@pytest.mark.parametrize(
    "strength,duration",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)],
)
def test_shake_rejects_nonpositive_params(strength, duration):
    with pytest.raises(ValueError):
        CameraShake(strength, duration, 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_shake_rejects_nonfinite_params(value):
    with pytest.raises(ValueError):
        CameraShake(value, 1.0, 1)
    with pytest.raises(ValueError):
        CameraShake(1.0, value, 1)
    with pytest.raises(ValueError):
        CameraShake(1.0, 1.0, 1).update(value)


def test_shake_rejects_bad_types():
    with pytest.raises(TypeError):
        CameraShake("1", 1.0, 1)
    with pytest.raises(TypeError):
        CameraShake(1.0, 1.0, 1.5)
    with pytest.raises(TypeError):
        CameraShake(1.0, 1.0, 1).update(None)


def test_shake_negative_dt_raises():
    with pytest.raises(ValueError):
        CameraShake(1.0, 1.0, 1).update(-1e-6)


def test_shake_100k_updates_under_budget():
    s = CameraShake(1.0, 1e9, 7)
    dt = 1 / 120.0
    s.update(dt)
    t0 = time.perf_counter()
    for _ in range(100000):
        s.update(dt)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 200.0, elapsed_ms


def test_shake_concurrent_updates_are_safe():
    s = CameraShake(1.0, 1e9, 11)
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            for _ in range(2000):
                dx, dy = s.update(1e-5)
                assert abs(dx) <= 1.0 and abs(dy) <= 1.0
        except BaseException as exc:  # pragma: no cover - 失败路径
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    assert s.remaining() > 0.0
