import math

import pytest

import uvscroll

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return uvscroll.UvScroll(0.5, 0.25, 4)


def _check_initial():
    return _s().uv() == (0.0, 0.0)


def _check_advance():
    s = _s()
    s.update(1.0)
    return abs(s.uv()[0] - 0.5) < 1e-9


def _check_wrap():
    s = _s()
    s.update(3.0)
    return abs(s.uv()[0] - 0.5) < 1e-9


def _check_negative():
    s = uvscroll.UvScroll(-0.5, 0.0, 1)
    s.update(0.5)
    return abs(s.uv()[0] - 0.75) < 1e-9


def _check_frame():
    s = _s()
    s.update(1.0)
    return 0 <= s.frame() < 4


def _check_bad_frames():
    return _raises(lambda: uvscroll.UvScroll(0.1, 0.1, 0))


def _check_negative_dt():
    return _raises(lambda: _s().update(-0.1))


def _check_type():
    return isinstance(_s().uv(), tuple)


def _check_range():
    s = _s()
    s.update(10.0)
    return 0.0 <= s.uv()[0] < 1.0


def _check_det():
    a = _s(); a.update(0.3)
    b = _s(); b.update(0.3)
    return a.uv() == b.uv()


def _check_zero_dt():
    s = _s()
    s.update(0.0)
    return s.uv() == (0.0, 0.0)


def _check_frames_advance():
    s = uvscroll.UvScroll(0.0, 0.0, 4)
    s.update(1.0)
    return 0 <= s.frame() <= 3


def _check_no_jump():
    s = _s()
    a = s.uv()[0]
    s.update(0.1)
    return s.uv()[0] >= a - 1e-9


def _check_float():
    return isinstance(_s().uv()[0], float)


def _check_no_state_leak():
    s = _s()
    for _ in range(10):
        s.update(0.5)
    return 0.0 <= s.uv()[0] < 1.0

CASES = [("initial", "_check_initial"), ("advance", "_check_advance"), ("wrap", "_check_wrap"), ("negative", "_check_negative"), ("frame", "_check_frame"), ("bad_frames", "_check_bad_frames"), ("negative_dt", "_check_negative_dt"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("zero_dt", "_check_zero_dt"), ("frames_advance", "_check_frames_advance"), ("no_jump", "_check_no_jump"), ("float", "_check_float"), ("no_state_leak", "_check_no_state_leak")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
