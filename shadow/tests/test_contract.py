"""shadow 回归测试：约束与边界（不改动既有 tests/test_core.py）。"""

import math
import threading
import time

import pytest

import shadow

SQ = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _inside(poly, point):
    x, y = point
    crossings = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y):
            xi = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xi:
                crossings += 1
    return crossings % 2 == 1


def test_shadow_polygon_contains_projected_vertices():
    for lz in (0.5, 1.0, 3.0, 50.0):
        out = shadow.Shadow((0.5, 0.5, lz)).project(SQ)
        assert len(out) == 4
        for vertex in SQ:
            assert _inside(out, vertex)


def test_shadow_area_decreases_monotonically_with_distance():
    areas = []
    for lz in (0.1, 1.0, 3.0, 10.0, 1e3, 1e6):
        s = shadow.Shadow((0.5, 0.5, lz))
        s.project(SQ)
        areas.append(s.area())
    assert all(b < a for a, b in zip(areas, areas[1:]))
    # 位似关系：(lz + 1) / lz 的平方
    s = shadow.Shadow((0.5, 0.5, 3.0))
    s.project(SQ)
    assert s.area() == pytest.approx((4.0 / 3.0) ** 2)


def test_shadow_light_inside_polygon_raises():
    with pytest.raises(ValueError):
        shadow.Shadow((0.5, 0.5, 0.0)).project(SQ)
    with pytest.raises(ValueError):
        shadow.Shadow((0.0, 0.5, 0.0)).project(SQ)  # 落在边界上


def test_shadow_degenerate_edges_have_no_duplicate_vertices():
    s = shadow.Shadow((0.5, 0.5, 3.0))
    out = s.project([(0, 0), (0, 0), (1, 0), (1, 1), (1, 1), (1, 1), (0, 1), (0, 0)])
    assert len(out) == 4
    assert len(set(out)) == len(out)


@pytest.mark.parametrize("light,exc", [
    ((0.0, 0.0, float("nan")), ValueError),
    ((0.0, float("inf"), 1.0), ValueError),
    ((0.0, 0.0, -1.0), ValueError),
    (("0", 0.0, 1.0), TypeError),
    (None, TypeError),
    ((0.0, 0.0), ValueError),
])
def test_shadow_bad_light_is_classified(light, exc):
    with pytest.raises(exc):
        shadow.Shadow(light)


@pytest.mark.parametrize("points,exc", [
    ([], ValueError),
    ([(0.0, float("nan"))], ValueError),
    ([(float("-inf"), 0.0)], ValueError),
    ([("x", 0.0)], TypeError),
    ([(0.0, 0.0, 0.0)], ValueError),
    (None, TypeError),
])
def test_shadow_bad_points_are_classified(points, exc):
    s = shadow.Shadow((0.5, 0.5, 3.0))
    with pytest.raises(exc):
        s.project(points)


def test_shadow_area_stays_finite_or_raises():
    s = shadow.Shadow((0.0, 0.0, 1e-8))
    with pytest.raises(ValueError):
        s.project([(1e200, 0.0), (0.0, 1e200), (0.0, 0.0)])


def test_shadow_performance_10k_vertices():
    points = [(math.cos(i) * 10.0, math.sin(i) * 10.0) for i in range(10000)]
    s = shadow.Shadow((0.0, 0.0, 5.0))
    start = time.perf_counter()
    out = s.project(points)
    elapsed = time.perf_counter() - start
    assert len(out) == 10000
    assert elapsed < 0.3


def test_shadow_concurrent_calls_are_consistent():
    s = shadow.Shadow((0.5, 0.5, 3.0))
    expected = [(0.5 + (x - 0.5) * 4.0 / 3.0, 0.5 + (y - 0.5) * 4.0 / 3.0)
                for x, y in SQ]
    errors = []

    def worker():
        for _ in range(100):
            if s.project(SQ) != expected or s.area() != (4.0 / 3.0) ** 2:
                errors.append(1)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
