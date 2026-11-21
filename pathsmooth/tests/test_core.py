import math

import pytest

import pathsmooth

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

LINE = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.0), (3.0, 0.5), (4.0, 0.0)]


def _s():
    return pathsmooth.PathSmoother()


def _check_endpoints():
    out = _s().smooth(LINE)
    return out[0] == LINE[0] and out[-1] == LINE[-1]


def _check_len():
    return len(_s().smooth(LINE)) == len(LINE)


def _check_move():
    out = _s().smooth(LINE)
    return any(out[i] != LINE[i] for i in range(1, len(LINE) - 1))


def _check_bad_points():
    return _raises(lambda: _s().smooth([(0.0, 0.0), (1.0, 1.0)]))


def _check_empty():
    return _raises(lambda: _s().smooth([]))


def _check_alpha():
    return _raises(lambda: pathsmooth.PathSmoother(1, 1.5))


def _check_iter():
    return _raises(lambda: pathsmooth.PathSmoother(0, 0.5))


def _check_det():
    return _s().smooth(LINE) == _s().smooth(LINE)


def _check_type():
    return isinstance(_s().smooth(LINE), list)


def _check_no_mut():
    src = list(LINE)
    _s().smooth(LINE)
    return LINE == src


def _check_bound():
    out = _s().smooth(LINE)
    return all(math.dist(a, b) <= 1.0 for a, b in zip(out, LINE))


def _check_more_iter():
    a = pathsmooth.PathSmoother(1, 0.5).smooth(LINE)
    b = pathsmooth.PathSmoother(5, 0.5).smooth(LINE)
    return a != b


def _check_flat():
    flat = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    out = _s().smooth(flat)
    return all(abs(p[1]) < 1e-9 for p in out)


def _check_float():
    return isinstance(_s().smooth(LINE)[1][0], float)


def _check_three():
    return len(_s().smooth([(0, 0), (1, 1), (2, 0)])) == 3

CASES = [("endpoints", "_check_endpoints"), ("len", "_check_len"), ("move", "_check_move"), ("bad_points", "_check_bad_points"), ("empty", "_check_empty"), ("alpha", "_check_alpha"), ("iter", "_check_iter"), ("det", "_check_det"), ("type", "_check_type"), ("no_mut", "_check_no_mut"), ("bound", "_check_bound"), ("more_iter", "_check_more_iter"), ("flat", "_check_flat"), ("float", "_check_float"), ("three", "_check_three")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
