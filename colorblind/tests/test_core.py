import math

import pytest

import colorblind

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c(kind="deuteranopia"):
    return colorblind.ColorBlind(kind)


def _check_white():
    out = _c().apply((1.0, 1.0, 1.0))
    return all(abs(v - 1.0) < 1e-6 for v in out)


def _check_black():
    return _c().apply((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def _check_gray():
    out = _c().apply((0.5, 0.5, 0.5))
    return abs(out[0] - 0.5) < 1e-6 and abs(out[1] - 0.5) < 1e-6


def _check_bad_kind():
    return _raises(lambda: colorblind.ColorBlind("bogus"))


def _check_type():
    return isinstance(_c().apply((1.0, 0.0, 0.0)), tuple)


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _c().apply((0.8, 0.2, 0.1)))


def _check_det():
    return _c().apply((0.3, 0.4, 0.5)) == _c().apply((0.3, 0.4, 0.5))


def _check_three_kinds():
    kinds = ["protanopia", "deuteranopia", "tritanopia"]
    return all(isinstance(colorblind.ColorBlind(k).apply((1.0, 0.0, 0.0)), tuple) for k in kinds)


def _check_image_shape():
    out = _c().apply_image([[(1.0, 1.0, 1.0)]])
    return len(out) == 1 and len(out[0]) == 1


def _check_image_empty():
    return _c().apply_image([]) == []


def _check_no_mut():
    img = [[(1.0, 1.0, 1.0)]]
    _c().apply_image(img)
    return img == [[(1.0, 1.0, 1.0)]]


def _check_sum_rows():
    out = _c().apply((1.0, 0.0, 0.0))
    return abs(sum(out) - 1.0) < 0.2


def _check_bad_color():
    return _raises(lambda: _c().apply((1.0, 1.0)))


def _check_float():
    return isinstance(_c().apply((0.2, 0.3, 0.4))[0], float)


def _check_red_changed():
    return _c().apply((1.0, 0.0, 0.0)) != (1.0, 0.0, 0.0)

CASES = [("white", "_check_white"), ("black", "_check_black"), ("gray", "_check_gray"), ("bad_kind", "_check_bad_kind"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("three_kinds", "_check_three_kinds"), ("image_shape", "_check_image_shape"), ("image_empty", "_check_image_empty"), ("no_mut", "_check_no_mut"), ("sum_rows", "_check_sum_rows"), ("bad_color", "_check_bad_color"), ("float", "_check_float"), ("red_changed", "_check_red_changed")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
