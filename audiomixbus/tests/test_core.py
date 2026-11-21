import math

import pytest

import audiomixbus

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _b(send=1.0):
    return audiomixbus.MixBus(send)


def _check_empty():
    return _b().mix() == []


def _check_single():
    b = _b()
    b.push([0.5])
    return b.mix() == [0.5]


def _check_sum():
    b = _b()
    b.push([0.5])
    b.push([0.25])
    return abs(b.mix()[0] - 0.75) < 1e-9


def _check_send_zero():
    b = _b(0.0)
    b.push([0.5])
    return b.mix() == [0.0]


def _check_send_half():
    b = _b(0.5)
    b.push([0.5])
    return abs(b.mix()[0] - 0.25) < 1e-9


def _check_clamp():
    b = _b()
    b.push([1.0])
    b.push([1.0])
    return b.mix()[0] <= 1.0


def _check_bad_send():
    return _raises(lambda: audiomixbus.MixBus(2.0))


def _check_len():
    b = _b()
    b.push([0.1, 0.2])
    return len(b.mix()) == 2


def _check_type():
    b = _b()
    b.push([0.1])
    return isinstance(b.mix(), list)


def _check_det():
    b1 = _b(); b1.push([0.3])
    b2 = _b(); b2.push([0.3])
    return b1.mix() == b2.mix()


def _check_empty_push():
    b = _b()
    b.push([])
    return b.mix() == []


def _check_no_mut():
    s = [0.5]
    b = _b()
    b.push(s)
    return s == [0.5]


def _check_reset():
    b = _b()
    b.push([0.5])
    b.reset()
    return b.mix() == []


def _check_range():
    b = _b()
    b.push([0.9])
    return all(-1.0 <= v <= 1.0 for v in b.mix())


def _check_multiple_frames():
    b = _b()
    b.push([0.1, 0.2])
    b.push([0.1, 0.2])
    return b.mix() == [0.2, 0.4]

CASES = [("empty", "_check_empty"), ("single", "_check_single"), ("sum", "_check_sum"), ("send_zero", "_check_send_zero"), ("send_half", "_check_send_half"), ("clamp", "_check_clamp"), ("bad_send", "_check_bad_send"), ("len", "_check_len"), ("type", "_check_type"), ("det", "_check_det"), ("empty_push", "_check_empty_push"), ("no_mut", "_check_no_mut"), ("reset", "_check_reset"), ("range", "_check_range"), ("multiple_frames", "_check_multiple_frames")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
