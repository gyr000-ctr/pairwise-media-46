import math

import pytest

import spline

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PTS = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]


def _s():
    return spline.Spline(PTS)


def _check_start():
    return abs(_s().sample(0.0)[0] - 0.0) < 1e-9


def _check_end():
    return abs(_s().sample(1.0)[0] - 3.0) < 1e-6


def _check_mid():
    p = _s().sample(0.5)
    return 0.0 <= p[0] <= 3.0


def _check_bad_points():
    return _raises(lambda: spline.Spline([(0.0, 0.0)]))


def _check_empty():
    return _raises(lambda: spline.Spline([]))


def _check_t_range():
    return _raises(lambda: _s().sample(1.5))


def _check_resample_nonempty():
    return len(_s().resample(0.5)) >= 2


def _check_resample_endpoints():
    out = _s().resample(0.5)
    return abs(out[0][0]) < 1e-6 and abs(out[-1][0] - 3.0) < 0.1


def _check_spacing():
    out = _s().resample(0.5)
    return all(math.dist(a, b) <= 1.0 for a, b in zip(out, out[1:]))


def _check_bad_spacing():
    return _raises(lambda: _s().resample(0.0))


def _check_type():
    return isinstance(_s().sample(0.2), tuple)


def _check_det():
    return _s().sample(0.3) == _s().sample(0.3)


def _check_no_mut():
    src = list(PTS)
    _s().sample(0.5)
    return PTS == src


def _check_two_points():
    return len(spline.Spline([(0.0, 0.0), (1.0, 0.0)]).resample(0.5)) >= 2


def _check_float():
    return isinstance(_s().sample(0.1)[0], float)

CASES = [("start", "_check_start"), ("end", "_check_end"), ("mid", "_check_mid"), ("bad_points", "_check_bad_points"), ("empty", "_check_empty"), ("t_range", "_check_t_range"), ("resample_nonempty", "_check_resample_nonempty"), ("resample_endpoints", "_check_resample_endpoints"), ("spacing", "_check_spacing"), ("bad_spacing", "_check_bad_spacing"), ("type", "_check_type"), ("det", "_check_det"), ("no_mut", "_check_no_mut"), ("two_points", "_check_two_points"), ("float", "_check_float")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
