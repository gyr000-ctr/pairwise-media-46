import math

import pytest

import hueshift

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _h(deg=90.0):
    return hueshift.HueShift(deg)


def _check_zero():
    return _h(0.0).apply((0.2, 0.4, 0.6)) == (0.2, 0.4, 0.6)


def _check_360():
    out = _h(360.0).apply((0.2, 0.4, 0.6))
    return all(abs(a - b) < 1e-6 for a, b in zip(out, (0.2, 0.4, 0.6)))


def _check_gray():
    out = _h().apply((0.5, 0.5, 0.5))
    return all(abs(v - 0.5) < 1e-6 for v in out)


def _check_bad_degrees():
    return _raises(lambda: hueshift.HueShift(400.0))


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _h().apply((0.9, 0.1, 0.1)))


def _check_type():
    return isinstance(_h().apply((0.3, 0.3, 0.3)), tuple)


def _check_det():
    return _h().apply((0.1, 0.2, 0.3)) == _h().apply((0.1, 0.2, 0.3))


def _check_changed():
    return _h().apply((1.0, 0.0, 0.0)) != (1.0, 0.0, 0.0)


def _check_bad_color():
    return _raises(lambda: _h().apply((1.0, 0.0)))


def _check_180():
    out = _h(180.0).apply((1.0, 0.0, 0.0))
    return abs(out[0]) < 0.1


def _check_negative_deg():
    return _raises(lambda: hueshift.HueShift(-1.0))


def _check_sum_approx():
    out = _h().apply((0.4, 0.4, 0.4))
    return abs(sum(out) - 1.2) < 1e-6


def _check_float():
    return isinstance(_h().apply((0.2, 0.3, 0.4))[0], float)


def _check_white():
    out = _h().apply((1.0, 1.0, 1.0))
    return all(abs(v - 1.0) < 1e-6 for v in out)


def _check_no_mut():
    c = (0.2, 0.4, 0.6)
    _h().apply(c)
    return c == (0.2, 0.4, 0.6)

CASES = [("zero", "_check_zero"), ("360", "_check_360"), ("gray", "_check_gray"), ("bad_degrees", "_check_bad_degrees"), ("range", "_check_range"), ("type", "_check_type"), ("det", "_check_det"), ("changed", "_check_changed"), ("bad_color", "_check_bad_color"), ("180", "_check_180"), ("negative_deg", "_check_negative_deg"), ("sum_approx", "_check_sum_approx"), ("float", "_check_float"), ("white", "_check_white"), ("no_mut", "_check_no_mut")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
