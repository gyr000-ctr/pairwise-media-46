"""uvscroll 边界契约自测：取模滚动、负速度、多层视差、帧/UV 同步、
循环接缝连续性、输入分类报错、性能上界与并发安全。

本文件不依赖、也不修改 ``tests/test_core.py`` 中的既有用例；
文件名中的 "scroll" 使其被验收命令 ``pytest tests -q -k scroll`` 选中。
"""

import math
import threading
import time

import pytest

import uvscroll
from uvscroll import UvScroll


def _circular_delta(a, b):
    """环 [0, 1) 上 a -> b 的最短有向距离。"""
    d = (b - a) % 1.0
    return d - 1.0 if d > 0.5 else d


# --------------------------------------------------------------------- 滚动
def test_scroll_accumulates_and_wraps_exactly():
    s = UvScroll(0.5, 0.25, 4)
    for _ in range(48):
        s.update(1.0)
        u, v = s.uv()
        assert 0.0 <= u < 1.0 and 0.0 <= v < 1.0
    # 48 秒 × 0.5 = 24.0 圈 → 相位为 0；v 为 12.0 圈 → 0
    assert s.uv() == pytest.approx((0.0, 0.0), abs=1e-9)


def test_scroll_fractional_wrap_matches_modulo():
    s = UvScroll(0.7, -0.3, 2)
    total_u = total_v = 0.0
    dt = 0.037
    for _ in range(500):
        s.update(dt)
        total_u += 0.7 * dt
        total_v += -0.3 * dt
    u, v = s.uv()
    assert u == pytest.approx(total_u % 1.0, abs=1e-9)
    assert v == pytest.approx(total_v % 1.0, abs=1e-9)


def test_scroll_negative_speed_is_correct():
    s = UvScroll(-0.25, -1.0, 1)
    s.update(1.0)
    assert s.uv() == pytest.approx((0.75, 0.0), abs=1e-12)
    s.update(2.0)  # 再退 0.5 圈
    assert s.uv() == pytest.approx((0.25, 0.0), abs=1e-12)
    # 任意步数后仍恒在 [0, 1)
    for _ in range(100):
        s.update(0.13)
        assert all(0.0 <= c < 1.0 for c in s.uv())


def test_scroll_frame_is_always_in_range():
    for frames in (1, 2, 3, 4, 7, 16):
        s = UvScroll(0.9, -0.4, frames)
        for _ in range(200):
            s.update(0.011)
            f = s.frame()
            assert isinstance(f, int) and 0 <= f < frames


# ----------------------------------------------------------------- 多层视差
def test_parallax_layers_have_distinct_speeds_and_stay_bounded():
    coeffs = [1.0, 0.5, 0.25, -0.5]
    s = UvScroll(0.6, 0.2, frames=8, layers=coeffs)
    assert s.layer_count() == len(coeffs)
    assert s.layer_coeffs() == tuple((c, c) for c in coeffs)
    s.update(1.0)
    for i, c in enumerate(coeffs):
        assert s.uv(i) == pytest.approx(((0.6 * c) % 1.0, (0.2 * c) % 1.0),
                                        abs=1e-12)
        assert s.speed(i) == pytest.approx((0.6 * c, 0.2 * c), abs=1e-12)


def test_parallax_accepts_pairs_and_layer_count_forms():
    s = UvScroll(0.5, 0.5, 4, layers=[(1.0, 0.0), (0.0, 1.0)])
    s.update(0.5)
    assert s.uv(0) == pytest.approx((0.25, 0.0), abs=1e-12)
    assert s.uv(1) == pytest.approx((0.0, 0.25), abs=1e-12)

    counted = UvScroll(0.5, 0.5, 4, layers=4)
    assert counted.layer_count() == 4
    assert counted.layer_coeffs()[0] == (1.0, 1.0)
    assert counted.layer_coeffs()[1] == (0.5, 0.5)


def test_parallax_uvs_and_frames_cover_every_layer():
    s = UvScroll(0.5, 0.25, frames=6, layers=[1.0, 0.5, 0.25])
    s.update(0.37)
    uvs = s.uvs()
    assert isinstance(uvs, tuple) and len(uvs) == 3
    assert s.layer_frames() == tuple(s.frame(i) for i in range(3))
    for (u, v), f in zip(uvs, s.layer_frames()):
        assert 0.0 <= u < 1.0 and 0.0 <= v < 1.0
        assert f == min(int(u * 6), 5)
    with pytest.raises(IndexError):
        s.uv(3)
    assert s.uv(-1) == s.uv(2)


def test_add_layer_keeps_existing_layers_untouched():
    s = UvScroll(0.5, 0.5, 4)
    s.update(0.2)
    before = s.uv()
    index = s.add_layer(0.5)
    assert index == 1
    assert s.uv(0) == before
    assert s.uv(1) == (0.0, 0.0)
    s.update(1.0)
    assert s.uv(1) == pytest.approx((0.25, 0.25), abs=1e-12)


# ------------------------------------------------------- 帧索引与 UV 同步
def test_frame_is_derived_from_uv_for_every_layer_and_step():
    frames = 10
    s = UvScroll(0.37, -0.21, frames, layers=[1.0, 0.5, 0.25, -1.0])
    for step in range(700):
        s.update(0.013 if step % 3 else 0.05)
        for i in range(s.layer_count()):
            u = s.uv(i)[0]
            expected = min(int(u * frames), frames - 1)
            assert s.frame(i) == expected, (step, i, u, s.frame(i), expected)


# ------------------------------------------------------------- 接缝连续性
def test_seam_uv_jump_bounded_by_step_displacement():
    speed = 0.8
    frames = 5
    for dt in (0.001, 0.016, 0.1, 0.333):
        s = UvScroll(speed, -speed, frames, layers=[1.0, 0.5])
        bound = speed * dt + 1e-12
        for _ in range(500):
            prev = s.uvs()
            s.update(dt)
            cur = s.uvs()
            for (pu, pv), (cu, cv) in zip(prev, cur):
                assert abs(_circular_delta(pu, cu)) <= bound
                assert abs(_circular_delta(pv, cv)) <= bound


def test_seam_frame_jump_is_at_most_one():
    frames = 4
    s = UvScroll(0.6, 0.6, frames, layers=[1.0, 0.5])
    for _ in range(2000):
        prev = s.layer_frames()
        s.update(0.01)
        for p, c in zip(prev, s.layer_frames()):
            step = (c - p) % frames
            assert step <= 1


# ---------------------------------------------------------------- 输入校验
@pytest.mark.parametrize("bad", [0, -1, -7])
def test_bad_frame_counts_raise_value_error(bad):
    with pytest.raises(ValueError):
        UvScroll(0.1, 0.1, bad)


@pytest.mark.parametrize("bad", [1.0, 2.5, "4", None, True])
def test_bad_frame_types_raise_type_error(bad):
    with pytest.raises(TypeError):
        UvScroll(0.1, 0.1, bad)


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_speeds_raise_value_error(speed):
    with pytest.raises(ValueError):
        UvScroll(speed, 0.0, 1)
    with pytest.raises(ValueError):
        UvScroll(0.0, speed, 1)


@pytest.mark.parametrize("speed", ["0.5", None, object(), True])
def test_bad_speed_types_raise_type_error(speed):
    with pytest.raises(TypeError):
        UvScroll(speed, 0.0, 1)


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_dt_raises_value_error(dt):
    with pytest.raises(ValueError):
        UvScroll(0.5, 0.5, 1).update(dt)


@pytest.mark.parametrize("dt", ["0.1", None, object(), True])
def test_bad_dt_types_raise_type_error(dt):
    with pytest.raises(TypeError):
        UvScroll(0.5, 0.5, 1).update(dt)


def test_negative_dt_raises_and_leaves_state_intact():
    s = UvScroll(0.5, 0.5, 1)
    s.update(0.3)
    before = s.uv()
    with pytest.raises(ValueError):
        s.update(-0.1)
    assert s.uv() == before


@pytest.mark.parametrize("layers", [0, -3, [], (), 5000, [0.5] * 5000])
def test_illegal_layer_counts_raise_value_error(layers):
    with pytest.raises(ValueError):
        UvScroll(0.5, 0.5, 1, layers=layers)


@pytest.mark.parametrize("layers", [[0.5, (1.0,)], [(1.0, 2.0, 3.0)],
                                    [(1.0, float("nan"))],
                                    [float("inf"), 0.5]])
def test_illegal_layer_coeffs_raise_value_error(layers):
    with pytest.raises(ValueError):
        UvScroll(0.5, 0.5, 1, layers=layers)


@pytest.mark.parametrize("layers", ["1,0.5", [None], [object()], True,
                                    [(1.0, "x")]])
def test_illegal_layer_types_raise_type_error(layers):
    with pytest.raises(TypeError):
        UvScroll(0.5, 0.5, 1, layers=layers)


def test_conflicting_layer_specs_raise():
    with pytest.raises(TypeError):
        UvScroll(0.5, 0.5, 1, layers=[0.5], parallax=[1.0])


def test_layer_index_requires_integer():
    s = UvScroll(0.5, 0.5, 1)
    with pytest.raises(TypeError):
        s.uv(0.5)
    with pytest.raises(TypeError):
        s.uv(True)


def test_nonfinite_is_distinguishable_from_bad_type():
    with pytest.raises(uvscroll.NonFiniteInputError):
        UvScroll(0.5, float("nan"), 1)
    assert issubclass(uvscroll.NonFiniteInputError, ValueError)
    # 类型错误同时满足 TypeError / ValueError 两种捕获约定
    with pytest.raises(TypeError):
        UvScroll("0.5", 0.5, 1)
    with pytest.raises(ValueError):
        UvScroll("0.5", 0.5, 1)
    try:
        UvScroll("0.5", 0.5, 1)
    except ValueError as exc:
        assert not isinstance(exc, uvscroll.NonFiniteInputError)


def test_overflowing_dt_is_rejected_without_state_change():
    s = UvScroll(1e308, 1e308, 4, layers=[1.0, 0.5])
    s.update(0.0)
    before = s.uvs()
    with pytest.raises(ValueError):
        s.update(1e300)
    assert s.uvs() == before


def test_speed_pair_and_layer_forms_are_equivalent():
    a = UvScroll(0.5, 0.25, 4, layers=[1.0, 0.5])
    b = UvScroll((0.5, 0.25), frames=4, parallax=[1.0, 0.5])
    a.update(0.8)
    b.update(0.8)
    assert a.uvs() == b.uvs()
    with pytest.raises(TypeError):
        UvScroll(0.5, 0.25, 4, layers=[1.0], parallax=[1.0])
    with pytest.raises(TypeError):
        UvScroll((0.5, 0.25), frames=4, speed=(0.5, 0.25))
    with pytest.raises(ValueError):
        UvScroll(0.5, 0.25, 4, layers=[1.0, 0.5], n_layers=3)
    with pytest.raises(TypeError):
        UvScroll(0.5, 0.25, 4, unknown_kwarg=1)


# ------------------------------------------------------------ 隔离与并发
def test_instances_do_not_share_state():
    a = UvScroll(0.5, 0.5, 4)
    b = UvScroll(0.5, 0.5, 4)
    a.update(0.4)
    assert b.uv() == (0.0, 0.0) and b.frame() == 0
    a.reset()
    assert a.uv() == (0.0, 0.0) and b.uv() == (0.0, 0.0)


def test_concurrent_updates_are_serialized_and_total_is_exact():
    threads, per_thread, dt, speed = 8, 1000, 0.001, 0.5
    s = UvScroll(speed, -speed, 4, layers=[1.0, 0.5])
    errors = []

    def worker():
        try:
            for _ in range(per_thread):
                s.update(dt)
                u, v = s.uv()
                assert 0.0 <= u < 1.0 and 0.0 <= v < 1.0
                assert 0 <= s.frame() < 4
        except BaseException as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not errors, errors

    total = threads * per_thread * dt
    # 无丢失/重复累积：相位等于总量取模（按环形度量比较，容许累加舍入）
    assert abs(_circular_delta((speed * total) % 1.0, s.uv(0)[0])) < 1e-9
    assert abs(_circular_delta((speed * 0.5 * total) % 1.0, s.uv(1)[0])) < 1e-9
    assert abs(_circular_delta((-speed * total) % 1.0, s.uv(0)[1])) < 1e-9


# ---------------------------------------------------------------- 性能上界
@pytest.mark.parametrize("layers", [None, [1.0, 0.5, 0.25]])
def test_100k_updates_under_100ms(layers):
    s = UvScroll(0.5, 0.25, 4, layers=layers)
    dt = 1.0 / 120.0
    s.update(dt)  # 预热
    start = time.perf_counter()
    for _ in range(100000):
        s.update(dt)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 100.0, elapsed_ms
    assert math.isfinite(s.uv()[0])
