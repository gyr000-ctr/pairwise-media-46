import math

import pytest

import shadow

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

SQ = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _s():
    return shadow.Shadow((0.5, 0.5, 3.0))


def _check_nonempty():
    return len(_s().project(SQ)) > 0


def _check_count():
    return len(_s().project(SQ)) >= 4


def _check_offscreen():
    out = _s().project(SQ)
    return all(abs(pt[0]) < 100.0 and abs(pt[1]) < 100.0 for pt in out)


def _check_light_inside():
    return _raises(lambda: shadow.Shadow((0.5, 0.5, 0.0)).project(SQ))


def _check_bad_light():
    return _raises(lambda: shadow.Shadow((0.0, 0.0, -1.0)))


def _check_degenerate():
    s = _s()
    return len(s.project([(0.0, 0.0), (0.0, 0.0)])) <= 2


def _check_area_positive():
    s = _s()
    s.project(SQ)
    return s.area() > 0.0


def _check_area_monotonic():
    a = shadow.Shadow((0.5, 0.5, 1.0))
    b = shadow.Shadow((0.5, 0.5, 10.0))
    a.project(SQ)
    b.project(SQ)
    return b.area() <= a.area() + 1e-9


def _check_type():
    return isinstance(_s().project(SQ), list)


def _check_det():
    return _s().project(SQ) == _s().project(SQ)


def _check_no_mutation():
    src = list(SQ)
    _s().project(SQ)
    return SQ == src


def _check_empty():
    return _raises(lambda: _s().project([]))


def _check_single():
    return len(_s().project([(0.0, 0.0)])) >= 1


def _check_float():
    return isinstance(_s().project(SQ)[0][0], float)


def _check_area_no_project():
    return _raises(lambda: shadow.Shadow((0.5, 0.5, 3.0)).area())

CASES = [("nonempty", "_check_nonempty"), ("count", "_check_count"), ("offscreen", "_check_offscreen"), ("light_inside", "_check_light_inside"), ("bad_light", "_check_bad_light"), ("degenerate", "_check_degenerate"), ("area_positive", "_check_area_positive"), ("area_monotonic", "_check_area_monotonic"), ("type", "_check_type"), ("det", "_check_det"), ("no_mutation", "_check_no_mutation"), ("empty", "_check_empty"), ("single", "_check_single"), ("float", "_check_float"), ("area_no_project", "_check_area_no_project")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
