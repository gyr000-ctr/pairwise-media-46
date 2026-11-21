import math

import pytest

import tilefade

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f():
    return tilefade.TileFade(1.0)


def _check_initial():
    return _f().alpha() == 0.0


def _check_half():
    f = _f()
    f.update(0.5)
    return abs(f.alpha() - 0.5) < 1e-9


def _check_full():
    f = _f()
    f.update(5.0)
    return f.alpha() == 1.0


def _check_monotonic():
    f = _f()
    vals = []
    for _ in range(5):
        f.update(0.1)
        vals.append(f.alpha())
    return all(vals[i] <= vals[i + 1] + 1e-9 for i in range(4))


def _check_bad_duration():
    return _raises(lambda: tilefade.TileFade(0.0))


def _check_negative_dt():
    return _raises(lambda: _f().update(-0.1))


def _check_type():
    return isinstance(_f().alpha(), float)


def _check_range():
    f = _f()
    f.update(0.7)
    return 0.0 <= f.alpha() <= 1.0


def _check_det():
    a = _f(); a.update(0.3)
    b = _f(); b.update(0.3)
    return a.alpha() == b.alpha()


def _check_independent():
    a = _f(); b = _f()
    a.update(0.5)
    return a.alpha() != b.alpha()


def _check_fade_out():
    f = tilefade.TileFade(1.0, fade_in=False)
    f.update(0.5)
    return abs(f.alpha() - 0.5) < 1e-9


def _check_done():
    f = _f()
    f.update(2.0)
    return f.done() is True


def _check_not_done():
    return _f().done() is False


def _check_zero_dt():
    return _f().update(0.0) == 0.0


def _check_no_jump():
    f = _f()
    prev = f.alpha()
    f.update(0.01)
    return abs(f.alpha() - prev) <= 0.01 + 1e-9

CASES = [("initial", "_check_initial"), ("half", "_check_half"), ("full", "_check_full"), ("monotonic", "_check_monotonic"), ("bad_duration", "_check_bad_duration"), ("negative_dt", "_check_negative_dt"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("independent", "_check_independent"), ("fade_out", "_check_fade_out"), ("done", "_check_done"), ("not_done", "_check_not_done"), ("zero_dt", "_check_zero_dt"), ("no_jump", "_check_no_jump")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
