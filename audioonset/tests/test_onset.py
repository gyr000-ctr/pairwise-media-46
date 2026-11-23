"""多频带起音检测的约束用例（新增，不改动既有测试）。

覆盖：能量包络 + 谱通量融合判定、迟滞、最小间隔、静音不触发、
性能上界（100 万样本 < 400 ms）、跨缓冲区状态连续、
非法参数与非有限样本可区分报错、并发安全。
"""

import threading
import time

import pytest

import audioonset
from audioonset import NonFiniteSampleError, OnsetDetector


def _step_signal(n, amplitude=0.5, every=0):
    """静音背景上的阶跃/方波信号。"""
    data = [0.0] * n
    if every:
        high = False
        for i in range(n):
            if i % every == 0:
                high = not high
            if high:
                data[i] = amplitude
    else:
        data[n // 2 :] = [amplitude] * (n - n // 2)
    return data


def test_onset_step_detected():
    det = OnsetDetector(1000, 0.1, 10)
    onsets = det.process(_step_signal(256))
    assert len(onsets) >= 1
    assert all(0 <= p < 256 for p in onsets)


def test_onset_tone_burst_detected():
    # 谱变化（正弦突发）同样触发：谱通量参与融合判定
    import math

    det = OnsetDetector(1000, 0.1, 10)
    data = [0.0] * 128 + [0.4 * math.sin(2 * math.pi * i / 16) for i in range(128)]
    assert len(det.process(data)) >= 1


def test_onset_silence_and_quiet():
    det = OnsetDetector(1000, 0.1, 10)
    assert det.process([0.0] * 512) == []
    assert det.process([0.01] * 512) == []  # 低于阈值不触发


def test_onset_hysteresis_and_min_gap():
    det = OnsetDetector(1000, 0.1, 50)
    onsets = det.process(_step_signal(2000, amplitude=0.5, every=30))
    assert len(onsets) >= 2
    assert all(b - a >= 50 for a, b in zip(onsets, onsets[1:]))


def test_onset_cross_buffer_consistent():
    data = _step_signal(4096, amplitude=0.5, every=257)
    whole = OnsetDetector(1000, 0.1, 10).process(data)
    for chunk in (7, 16, 100, 1000):
        det = OnsetDetector(1000, 0.1, 10)
        got = []
        for i in range(0, len(data), chunk):
            got.extend(det.process(data[i : i + chunk]))
        assert got == whole


def test_onset_performance_1m():
    data = _step_signal(1_000_000, amplitude=0.5, every=50_000)
    det = OnsetDetector(44100, 0.1, 100)
    t0 = time.perf_counter()
    onsets = det.process(data)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.4, f"100 万样本耗时 {elapsed:.3f}s 超过 400ms"
    assert len(onsets) >= 10


def test_onset_invalid_params():
    for bad in (0, -1, float("nan"), float("inf"), "x", None):
        with pytest.raises(ValueError):
            OnsetDetector(bad, 0.1, 10)
        with pytest.raises(ValueError):
            OnsetDetector(1000, bad, 10)
        with pytest.raises(ValueError):
            OnsetDetector(1000, 0.1, bad)


def test_onset_nonfinite_distinguishable():
    det = OnsetDetector(1000, 0.1, 10)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(NonFiniteSampleError):
            det.process([0.0, bad])
    # 与参数错误可区分：类型不同、消息不同
    try:
        det.process([float("nan")])
    except ValueError as exc:
        assert type(exc) is NonFiniteSampleError
        assert "NaN" in str(exc) or "无穷" in str(exc)
    try:
        OnsetDetector(0, 0.1, 10)
    except ValueError as exc:
        assert type(exc) is ValueError
        assert "非法参数" in str(exc)


def test_onset_concurrent_safe():
    data = _step_signal(4096, amplitude=0.5, every=128)
    expected = OnsetDetector(1000, 0.1, 10).process(data)
    errors = []

    def worker():
        try:
            for _ in range(20):
                assert OnsetDetector(1000, 0.1, 10).process(data) == expected
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
