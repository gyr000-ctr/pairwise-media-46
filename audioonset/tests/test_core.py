import math

import pytest

import audioonset

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _o():
    return audioonset.OnsetDetector(1000, 0.1, 10)


def _check_silence():
    return _o().process([0.0] * 100) == []


def _check_single():
    data = [0.0] * 50 + [0.5] * 50
    return len(_o().process(data)) >= 1


def _check_len_type():
    return isinstance(_o().process([0.0] * 10), list)


def _check_bad_threshold():
    return _raises(lambda: audioonset.OnsetDetector(1000, 0.0, 10))


def _check_bad_gap():
    return _raises(lambda: audioonset.OnsetDetector(1000, 0.1, 0))


def _check_bad_rate():
    return _raises(lambda: audioonset.OnsetDetector(0, 0.1, 10))


def _check_positions_range():
    data = [0.0] * 20 + [0.5] * 80
    ons = _o().process(data)
    return all(0 <= p < len(data) for p in ons)


def _check_min_gap():
    data = ([0.5] * 5 + [0.0] * 5) * 20
    ons = _o().process(data)
    return all(b - a >= 10 for a, b in zip(ons, ons[1:]))


def _check_det():
    data = [0.0] * 30 + [0.5] * 30
    return _o().process(data) == _o().process(data)


def _check_empty():
    return _o().process([]) == []


def _check_quiet_no_trigger():
    return _o().process([0.01] * 100) == []


def _check_nan():
    return _raises(lambda: _o().process([float("nan")]))


def _check_state():
    o = _o()
    o.process([0.0] * 10)
    return isinstance(o.process([0.5] * 10), list)


def _check_threshold_effect():
    data = [0.0] * 20 + [0.2] * 80
    a = audioonset.OnsetDetector(1000, 0.1, 10).process(data)
    b = audioonset.OnsetDetector(1000, 0.5, 10).process(data)
    return len(a) >= len(b)


def _check_no_mutation():
    data = [0.0] * 10
    _o().process(data)
    return data == [0.0] * 10

CASES = [("silence", "_check_silence"), ("single", "_check_single"), ("len_type", "_check_len_type"), ("bad_threshold", "_check_bad_threshold"), ("bad_gap", "_check_bad_gap"), ("bad_rate", "_check_bad_rate"), ("positions_range", "_check_positions_range"), ("min_gap", "_check_min_gap"), ("det", "_check_det"), ("empty", "_check_empty"), ("quiet_no_trigger", "_check_quiet_no_trigger"), ("nan", "_check_nan"), ("state", "_check_state"), ("threshold_effect", "_check_threshold_effect"), ("no_mutation", "_check_no_mutation")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
