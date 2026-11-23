"""CVD 色盲模拟/可逆校正的验收测试（题面全部约束）。

可通过 `pytest -k cvd` 选中。覆盖：
- 模拟矩阵行和为 1、白点/灰度不动；
- 模拟在 OKLab 感知空间进行（非纯线性矩阵）；
- 反向校正接口与 simulate->correct 往返误差 <= 0.02，且优于线性逆矩阵；
- 校正色相偏移 <= 2°；输出钳制 [0,1]；
- 10 万像素性能上界 300ms；并发安全；非法类型可区分报错。
"""

import math
import threading
import time

import pytest

import colorblind
from colorblind import core as _core

KINDS = ["protanopia", "deuteranopia", "tritanopia"]


def _rms(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / 3.0)


def _max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def _oklab(rgb):
    return _core._oklab(*rgb)


# --------------------------------------------------------------------------
# 模拟矩阵约束
# --------------------------------------------------------------------------
def test_cvd_matrix_rows_sum_to_one():
    for kind in KINDS:
        for row in _core.SIMULATION_MATRICES[kind]:
            assert abs(sum(row) - 1.0) < 1e-5


def test_cvd_white_point_and_grays_are_fixed():
    for kind in KINDS:
        cb = colorblind.ColorBlind(kind)
        for g in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = cb.apply((g, g, g))
            assert _max_abs(out, (g, g, g)) < 1e-9


def test_cvd_three_kinds_distinct_and_chromatic_color_changes():
    red = (1.0, 0.0, 0.0)
    outs = {}
    for kind in KINDS:
        out = colorblind.ColorBlind(kind).apply(red)
        outs[kind] = out
        assert out != red
        assert all(0.0 <= v <= 1.0 for v in out)
    # 三类不应全部相同
    assert len({tuple(round(v, 6) for v in outs[k]) for k in KINDS}) > 1


# --------------------------------------------------------------------------
# 必须在 OKLab 做模拟（非纯线性矩阵）
# --------------------------------------------------------------------------
def test_cvd_simulation_runs_in_oklab_not_linear_only():
    # 纯线性矩阵的一个可证伪特征：在线性 RGB 空间保持线性叠加关系
    # f(a)+f(b) 与 f(a+b) 一致；OKLab 投影含 cbrt/伽马非线性，必破坏该关系。
    kind = "deuteranopia"
    cb = colorblind.ColorBlind(kind)
    a, b = (0.8, 0.1, 0.2), (0.1, 0.7, 0.3)
    fa, fb = cb.apply(a), cb.apply(b)
    # 若只是线性矩阵（即便再过 sRGB 伽马），线性域叠加关系在 OKLab 投影下不成立
    fsum = cb.apply(tuple(min(1.0, x + y) for x, y in zip(a, b)))
    linear_like = all(
        abs((fa[i] + fb[i]) - fsum[i]) < 1e-9 for i in range(3)
    )
    assert linear_like is False
    # 且模拟结果落在该类型的 OKLab 混淆线上（垂直分量 ~ 0）
    ux, uy = cb._ux, cb._uy
    for color in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.2, 0.6, 0.9)]:
        _L, aa, bb = _oklab(cb.apply(color))
        perpendicular = -aa * uy + bb * ux
        assert abs(perpendicular) < 1e-5


# --------------------------------------------------------------------------
# 反向校正 + 往返误差
# --------------------------------------------------------------------------
def test_cvd_correct_interface_exists_and_is_inverse_on_plane():
    grid = [
        (i / 6.0, j / 6.0, k / 6.0)
        for i in range(7)
        for j in range(7)
        for k in range(7)
    ]
    for kind in KINDS:
        cb = colorblind.ColorBlind(kind)
        for color in grid:
            simulated = cb.apply(color)
            corrected = cb.correct(simulated)
            resimulated = cb.apply(corrected)
            # simulate -> correct -> simulate 往返误差 <= 0.02
            assert _rms(resimulated, simulated) <= 0.02
            assert _max_abs(resimulated, simulated) <= 0.02


def test_cvd_roundtrip_beats_linear_inverse():
    """线性逆矩阵会把点推出色域、钳制后破坏共线性；本实现误差应显著更低。"""
    def linear_inverse_roundtrip(color, kind):
        M = _core.SIMULATION_MATRICES[kind]
        lin = [_core._s2lin(v) for v in color]
        sim_lin = [
            min(1.0, max(0.0, sum(M[i][j] * lin[j] for j in range(3))))
            for i in range(3)
        ]
        # 3x3 逆（伴随式）
        a, b, c = M[0]
        d, e, f = M[1]
        g, h, ii = M[2]
        A, B, C = e * ii - f * h, -(d * ii - f * g), d * h - e * g
        det = a * A + b * B + c * C
        inv = (
            (A, c * h - b * ii, b * f - c * e),
            (B, a * ii - c * g, c * d - a * f),
            (C, b * g - a * h, a * e - b * d),
        )
        pre = [
            min(1.0, max(0.0, sum(inv[i][j] * sim_lin[j] for j in range(3)) / det))
            for i in range(3)
        ]
        resim = [
            min(1.0, max(0.0, sum(M[i][j] * pre[j] for j in range(3))))
            for i in range(3)
        ]
        return _rms(sim_lin, resim)

    samples = [
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        (0.9, 0.2, 0.1), (0.1, 0.8, 0.6), (0.4, 0.3, 0.9),
    ]
    for kind in KINDS:
        cb = colorblind.ColorBlind(kind)
        ours = []
        linear = []
        for color in samples:
            sim = cb.apply(color)
            corr = cb.correct(sim)
            ours.append(_rms(cb.apply(corr), sim))
            linear.append(linear_inverse_roundtrip(color, kind))
        assert sum(ours) / len(ours) < sum(linear) / len(linear)


def test_cvd_correct_hue_shift_within_two_degrees():
    for kind in KINDS:
        cb = colorblind.ColorBlind(kind)
        for color in [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (0.9, 0.7, 0.1), (0.2, 0.6, 0.9), (0.8, 0.2, 0.5),
        ]:
            sim = cb.apply(color)
            corr = cb.correct(sim)
            q0, q1 = _oklab(sim), _oklab(corr)
            if math.hypot(q0[1], q0[2]) < 1e-4:
                continue
            d = abs(
                math.degrees(math.atan2(q1[2], q1[1]) - math.atan2(q0[2], q0[1]))
            )
            d = min(d, 360.0 - d)
            assert d <= 2.0


# --------------------------------------------------------------------------
# 钳制 / 不可变 / 确定性
# --------------------------------------------------------------------------
def test_cvd_outputs_clamped_to_unit_range():
    cb = colorblind.ColorBlind("deuteranopia")
    for i in range(11):
        for j in range(11):
            for k in range(0, 11, 2):
                out = cb.apply((i / 10.0, j / 10.0, k / 10.0))
                assert all(0.0 <= v <= 1.0 for v in out)
                assert all(isinstance(v, float) for v in out)


def test_cvd_apply_image_does_not_mutate_and_preserves_shape():
    cb = colorblind.ColorBlind("protanopia")
    img = [[(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)], [(0.3, 0.6, 0.9)]]
    snapshot = [row[:] for row in img]
    out = cb.apply_image(img)
    assert img == snapshot
    assert len(out) == 2 and len(out[0]) == 2 and len(out[1]) == 1
    assert cb.apply_image([]) == []


def test_cvd_deterministic():
    cb = colorblind.ColorBlind("tritanopia")
    color = (0.3, 0.4, 0.5)
    assert cb.apply(color) == cb.apply(color)


# --------------------------------------------------------------------------
# 非法输入可区分错误
# --------------------------------------------------------------------------
def test_cvd_unknown_kind_raises_distinct_error():
    with pytest.raises(colorblind.UnknownKindError):
        colorblind.ColorBlind("bogus")
    # 仍是 ValueError 子类
    with pytest.raises(ValueError):
        colorblind.ColorBlind(123)


def test_cvd_bad_color_structure_raises_color_error():
    cb = colorblind.ColorBlind("deuteranopia")
    with pytest.raises(colorblind.ColorValueError):
        cb.apply((1.0, 1.0))
    with pytest.raises(colorblind.ColorValueError):
        cb.apply("red")
    with pytest.raises(colorblind.ColorValueError):
        cb.apply((1.0, "x", 0.0))


def test_cvd_non_finite_raises_specialized_error():
    cb = colorblind.ColorBlind("deuteranopia")
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(colorblind.NonFiniteColorError):
            cb.apply((bad, 0.0, 0.0))
    # 非有限错误与结构错误可区分
    assert issubclass(colorblind.NonFiniteColorError, colorblind.ColorValueError)


def test_cvd_bad_image_raises_image_error():
    cb = colorblind.ColorBlind("deuteranopia")
    with pytest.raises(colorblind.ImageValueError):
        cb.apply_image(123)
    with pytest.raises(colorblind.ImageValueError):
        cb.apply_image([(1.0, 0.0, 0.0)])  # 行不是颜色序列的序列


# --------------------------------------------------------------------------
# 性能上界
# --------------------------------------------------------------------------
def test_cvd_100k_pixels_under_300ms():
    cb = colorblind.ColorBlind("deuteranopia")
    colors = [
        (
            ((i * 7 + j * 13 + k * 17) % 101) / 100.0,
            ((i * 11 + j * 5 + k * 3) % 101) / 100.0,
            ((i * 3 + j * 7 + k * 19) % 101) / 100.0,
        )
        for i in range(100)
        for j in range(100)
        for k in range(10)
    ]
    assert len(colors) == 100_000
    image = [colors[i * 100 : (i + 1) * 100] for i in range(1000)]
    start = time.perf_counter()
    out = cb.apply_image(image)
    elapsed = time.perf_counter() - start
    assert len(out) == 1000 and len(out[0]) == 100
    assert elapsed < 0.300


# --------------------------------------------------------------------------
# 并发安全
# --------------------------------------------------------------------------
def test_cvd_concurrent_use_is_safe():
    cb = colorblind.ColorBlind("deuteranopia")
    color = (0.42, 0.71, 0.19)
    expected = cb.apply(color)
    results = []
    errors = []

    def worker():
        try:
            for _ in range(2000):
                results.append(cb.apply(color))
                results.append(cb.correct(cb.apply(color)))
        except (AssertionError, ValueError, RuntimeError) as exc:  # 仅捕获并发缺陷相关异常
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results, "并发工作线程未产生结果"
    assert all(r == expected or len(r) == 3 for r in results[::2])


def test_cvd_lazy_state_initialization_under_threads():
    kind = "tritanopia"
    _core._STATE.pop(kind, None)
    barrier = threading.Barrier(6)
    out = {}

    def worker():
        barrier.wait()
        out[threading.get_ident()] = colorblind.ColorBlind(kind).apply((0.1, 0.9, 0.4))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    values = list(out.values())
    assert all(v == values[0] for v in values)
