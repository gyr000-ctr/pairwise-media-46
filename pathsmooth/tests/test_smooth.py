import math
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import pathsmooth

LINE = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0), (3.0, 0.5), (4.0, 0.0)]
BOUND = 1.0


def _big_path(n=10_000):
    return [(float(i), float((i % 7) - 3) * 0.25) for i in range(n)]


def test_smooth_locks_endpoints():
    out = pathsmooth.PathSmoother().smooth(LINE)
    assert out[0] == (0.0, 0.0)
    assert out[-1] == (4.0, 0.0)


def test_smooth_moves_interior_and_keeps_length():
    out = pathsmooth.PathSmoother().smooth(LINE)
    assert len(out) == len(LINE)
    assert any(out[i] != LINE[i] for i in range(1, len(LINE) - 1))


def test_smooth_respects_displacement_bound():
    out = pathsmooth.PathSmoother(9, 1.0).smooth(LINE)
    assert all(math.dist(a, b) <= BOUND for a, b in zip(out, LINE))


def test_smooth_respects_explicit_bound():
    bound = 0.1
    out = pathsmooth.PathSmoother(5, 1.0, max_displacement=bound).smooth(LINE)
    assert all(math.dist(a, b) <= bound + 1e-12 for a, b in zip(out, LINE))


def test_smooth_converges_to_flat_line():
    out = pathsmooth.PathSmoother(500, 0.5).smooth(LINE)
    assert all(abs(y) < 1e-9 for _, y in out)
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in out)


def test_smooth_extra_iterations_change_result():
    a = pathsmooth.PathSmoother(1, 0.5).smooth(LINE)
    b = pathsmooth.PathSmoother(5, 0.5).smooth(LINE)
    assert a != b


def test_smooth_rejects_too_few_points():
    for bad in ([], [(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]):
        with pytest.raises(ValueError):
            pathsmooth.PathSmoother().smooth(bad)


def test_smooth_rejects_non_finite_and_malformed():
    bad_paths = [
        [(0.0, 0.0), (float("nan"), 1.0), (2.0, 0.0)],
        [(0.0, 0.0), (1.0, float("inf")), (2.0, 0.0)],
        [(0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 0.0)],
        [(0.0, 0.0), ("x", 1.0), (2.0, 0.0)],
    ]
    for bad in bad_paths:
        with pytest.raises(ValueError):
            pathsmooth.PathSmoother().smooth(bad)


def test_smooth_rejects_bad_config():
    with pytest.raises(ValueError):
        pathsmooth.PathSmoother(0, 0.5)
    with pytest.raises(ValueError):
        pathsmooth.PathSmoother(1, 1.5)
    with pytest.raises(ValueError):
        pathsmooth.PathSmoother(1, float("nan"))


def test_smooth_does_not_mutate_input():
    src = list(LINE)
    pathsmooth.PathSmoother().smooth(LINE)
    assert LINE == src


def test_smooth_is_deterministic_and_float():
    out = pathsmooth.PathSmoother().smooth(LINE)
    assert out == pathsmooth.PathSmoother().smooth(LINE)
    assert isinstance(out, list)
    assert all(isinstance(x, float) and isinstance(y, float) for x, y in out)


def test_smooth_three_points():
    out = pathsmooth.PathSmoother().smooth([(0, 0), (1, 1), (2, 0)])
    assert len(out) == 3
    assert out[0] == (0.0, 0.0) and out[-1] == (2.0, 0.0)


def test_smooth_thread_safe_shared_instance():
    smoother = pathsmooth.PathSmoother(3, 0.5)
    path = _big_path(200)
    expected = smoother.smooth(path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: smoother.smooth(path), range(64)))

    assert all(result == expected for result in results)


def test_smooth_10k_points_under_300ms():
    smoother = pathsmooth.PathSmoother()
    path = _big_path()
    best = min(_timed(smoother, path) for _ in range(3))
    assert best < 0.300


def _timed(smoother, path):
    start = time.perf_counter()
    out = smoother.smooth(path)
    elapsed = time.perf_counter() - start
    assert len(out) == len(path)
    return elapsed
