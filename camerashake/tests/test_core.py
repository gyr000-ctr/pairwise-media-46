import math

import pytest

import camerashake

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return camerashake.CameraShake(1.0, 1.0, 7)


def _check_zero_time():
    return _s().update(0.0) == (0.0, 0.0)


def _check_amplitude():
    out = _s().update(0.1)
    return abs(out[0]) <= 1.0 and abs(out[1]) <= 1.0


def _check_decay():
    s = _s()
    a = s.remaining()
    s.update(0.5)
    return s.remaining() < a


def _check_end_zero():
    s = _s()
    s.update(2.0)
    return s.remaining() == 0.0 and s.update(0.1) == (0.0, 0.0)


def _check_bad_strength():
    return _raises(lambda: camerashake.CameraShake(0.0, 1.0, 1))


def _check_bad_duration():
    return _raises(lambda: camerashake.CameraShake(1.0, 0.0, 1))


def _check_det():
    a = _s()
    b = _s()
    return a.update(0.2) == b.update(0.2)


def _check_seed_differs():
    a = camerashake.CameraShake(1.0, 1.0, 1)
    b = camerashake.CameraShake(1.0, 1.0, 2)
    return a.update(0.2) != b.update(0.2)


def _check_type():
    return isinstance(_s().update(0.1), tuple)


def _check_remaining_range():
    return 0.0 <= _s().remaining() <= 1.0


def _check_monotonic_decay():
    s = _s()
    vals = []
    for _ in range(5):
        s.update(0.1)
        vals.append(s.remaining())
    return all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))


def _check_negative_dt():
    return _raises(lambda: _s().update(-0.1))


def _check_float():
    return isinstance(_s().update(0.1)[0], float)


def _check_no_state_leak():
    s = _s()
    for _ in range(20):
        s.update(0.1)
    return s.remaining() == 0.0


def _check_duration_positive():
    return camerashake.CameraShake(1.0, 2.0, 3).remaining() > 0.0

CASES = [("zero_time", "_check_zero_time"), ("amplitude", "_check_amplitude"), ("decay", "_check_decay"), ("end_zero", "_check_end_zero"), ("bad_strength", "_check_bad_strength"), ("bad_duration", "_check_bad_duration"), ("det", "_check_det"), ("seed_differs", "_check_seed_differs"), ("type", "_check_type"), ("remaining_range", "_check_remaining_range"), ("monotonic_decay", "_check_monotonic_decay"), ("negative_dt", "_check_negative_dt"), ("float", "_check_float"), ("no_state_leak", "_check_no_state_leak"), ("duration_positive", "_check_duration_positive")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
