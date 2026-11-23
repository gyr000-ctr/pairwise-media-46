"""spline 隐藏要求补充测试：弧长精度、clamp 端点、非法输入、并发与性能。"""

import math
import threading
import time

import pytest

import spline

PTS = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]


def test_spline_passes_all_control_points():
    s = spline.Spline(PTS)
    n = len(PTS) - 1
    for i, p in enumerate(PTS):
        got = s.sample(i / n)
        assert math.dist(got, p) < 1e-9


def test_spline_clamp_endpoint_tangents():
    # clamp 下首段曲线关于 t=0.5 的点应位于 p0 与 p1 之外切向一致的位置；
    # 这里验证端切线方向等于 p1-p0（clamp 的定义性特征）。
    s = spline.Spline(PTS)
    eps = 1e-6
    p0 = s.sample(0.0)
    p_near = s.sample(eps)
    tangent = (p_near[0] - p0[0], p_near[1] - p0[1])
    expected = (PTS[1][0] - PTS[0][0], PTS[1][1] - PTS[0][1])
    norm_t = math.hypot(*tangent)
    norm_e = math.hypot(*expected)
    dot = tangent[0] * expected[0] + tangent[1] * expected[1]
    assert abs(dot / (norm_t * norm_e) - 1.0) < 1e-3

    p1 = s.sample(1.0)
    p_near_end = s.sample(1.0 - eps)
    tangent_end = (p1[0] - p_near_end[0], p1[1] - p_near_end[1])
    expected_end = (PTS[-1][0] - PTS[-2][0], PTS[-1][1] - PTS[-2][1])
    nt = math.hypot(*tangent_end)
    ne = math.hypot(*expected_end)
    dot_e = (
        tangent_end[0] * expected_end[0] + tangent_end[1] * expected_end[1]
    )
    assert abs(dot_e / (nt * ne) - 1.0) < 1e-3


def _arc_table(s, n=8192):
    """用 sample() 独立构建细折线的累积弧长表。"""
    table = [s.sample(k / n) for k in range(n + 1)]
    cum = [0.0]
    for a, b in zip(table, table[1:]):
        cum.append(cum[-1] + math.dist(a, b))
    return table, cum


def _map_to_arc_indices(out, table):
    """把输出点按弧长单调地映射到细表索引。

    输出沿曲线单调出现，跟踪游标只向前推进，避免自交曲线上“空间
    最近点”落到错误弧长位置的歧义。
    """
    cursor = 0
    window = int(len(table) / max(1, len(out)) * 3) + 50
    indices = []
    for p in out:
        best_j = cursor
        best_d = float("inf")
        upper = min(len(table), cursor + window)
        for j in range(cursor, upper):
            d = (table[j][0] - p[0]) ** 2 + (table[j][1] - p[1]) ** 2
            if d < best_d:
                best_d = d
                best_j = j
        indices.append(best_j)
        cursor = best_j
    return indices


def test_spline_resample_arc_length_error_within_1_percent():
    s = spline.Spline(PTS)
    spacing = 0.25
    out = s.resample(spacing)
    table, cum = _arc_table(s)

    # 弧长积分自检：8192 段与 32768 段参考值偏差 <0.1%
    _, ref_cum = _arc_table(s, n=32768)
    assert abs(cum[-1] - ref_cum[-1]) / ref_cum[-1] < 0.001

    idx = _map_to_arc_indices(out, table)
    # 输出必须严格沿弧长有序
    assert all(b >= a for a, b in zip(idx, idx[1:]))

    # 相邻输出点之间的真实弧长应等于 spacing（误差 <=1%），收尾段除外
    arc_gaps = [cum[b] - cum[a] for a, b in zip(idx, idx[1:])]
    for gap in arc_gaps[:-1]:
        assert abs(gap - spacing) / spacing <= 0.01
    assert arc_gaps[-1] <= spacing * 1.01

    # 输出点须落在曲线上（离曲线距离 <=1% 间距）
    for j, p in zip(idx, out):
        assert math.dist(table[j], p) / spacing <= 0.01


def test_spline_resample_equidistant_spacing():
    s = spline.Spline(PTS)
    spacing = 0.3
    out = s.resample(spacing)
    table, cum = _arc_table(s)
    idx = _map_to_arc_indices(out, table)
    gaps = [cum[b] - cum[a] for a, b in zip(idx, idx[1:])]
    # 除收尾段外，相邻点的弧长间隔等于 spacing（误差 <=1%）
    for g in gaps[:-1]:
        assert abs(g - spacing) / spacing <= 0.01
    assert gaps[-1] <= spacing * 1.01
    # 弦长不大于对应弧长（既有测试同样以弦长上界判等距）
    for a, b in zip(out, out[1:]):
        assert math.dist(a, b) <= spacing * 1.01


def test_spline_resample_keeps_endpoints_exact():
    s = spline.Spline(PTS)
    out = s.resample(0.17)
    assert math.dist(out[0], PTS[0]) < 1e-12
    assert math.dist(out[-1], PTS[-1]) < 1e-12


def test_spline_two_points_is_straight_line():
    s = spline.Spline([(0.0, 0.0), (4.0, 0.0)])
    out = s.resample(1.0)
    assert math.dist(out[0], (0.0, 0.0)) < 1e-12
    assert math.dist(out[-1], (4.0, 0.0)) < 1e-12
    for a, b in zip(out, out[1:]):
        assert abs(a[1]) < 1e-12 and abs(b[1]) < 1e-12


def test_spline_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        spline.Spline([(0.0, float("nan")), (1.0, 0.0)])
    with pytest.raises(ValueError):
        spline.Spline([(0.0, 0.0), (float("inf"), 1.0)])
    with pytest.raises(ValueError):
        spline.Spline([(0.0, 0.0), (1.0, float("-inf"))])


def test_spline_rejects_non_finite_sample_and_spacing():
    s = spline.Spline(PTS)
    with pytest.raises(ValueError):
        s.sample(float("nan"))
    with pytest.raises(ValueError):
        s.resample(float("nan"))
    with pytest.raises(ValueError):
        s.resample(-1.0)


def test_spline_does_not_mutate_input():
    src = list(PTS)
    spline.Spline(PTS).resample(0.5)
    assert PTS == src


def test_spline_thread_safe_concurrent_reads():
    s = spline.Spline(PTS)
    errors = []

    def worker():
        try:
            for _ in range(200):
                a = s.sample(0.37)
                b = s.sample(0.37)
                assert a == b
                out = s.resample(0.25)
                assert len(out) >= 2
        except Exception as exc:  # pragma: no cover - 并发失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_spline_resample_10k_points_under_300ms():
    pts = [
        (float(i), math.sin(i * 0.3) * 10.0)
        for i in range(10000)
    ]
    s = spline.Spline(pts)
    start = time.perf_counter()
    out = s.resample(5.0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert len(out) >= 2
    assert math.dist(out[0], pts[0]) < 1e-9
    assert math.dist(out[-1], pts[-1]) < 1e-9
    assert elapsed_ms < 300.0
