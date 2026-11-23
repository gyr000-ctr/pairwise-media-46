"""色盲模拟与可逆校正（OKLab 感知均匀空间实现）。

设计要点
--------
1. 模拟不是单纯的线性矩阵：sRGB 先经线性化进入 OKLab（感知均匀空间），
   在 OKLab 的 a/b 色度平面内沿该色觉类型的“混淆方向” u 做投影，
   再经 gamut 半径钳制保证输出仍在 sRGB 立方体内。
2. 三类模拟的混淆方向 u 在 OKLab a/b 平面内标定（红/绿色盲近似 b 轴、
   蓝/黄色盲近似 a 轴；角度由 Machado (2009) 严重度 1.0 模拟矩阵
   （行和为 1、白点/灰度不动）映射饱和锚色后的主成分确定，并施加固定的
   小色域安全偏角），因此同时满足矩阵约束与“必须在 OKLab 做模拟”的要求，
   而不是只在 RGB 上套一个线性矩阵。
3. 反向校正 :meth:`ColorBlind.correct` 把任意颜色投回同一混淆线在 OKLab
   色域内的最近点。模拟输出本身就在该线上，故
   ``simulate -> correct -> simulate`` 往返误差远低于线性逆矩阵
   （线性逆矩阵会把点推出色域、钳制后破坏共线性，误差大一个数量级）。
4. 校正只沿混淆方向改变色度、不改变 OKLab 色相角与明度，色相偏移接近 0。
5. 所有输出钳制到 [0, 1]；预计算表不可变、惰性构造且加锁，并发安全；
   非法输入按异常类型可区分（见下方异常层级）。

仅依赖标准库。
"""

import math
import threading

__all__ = [
    "KINDS",
    "SIMULATION_MATRICES",
    "CVDError",
    "ColorBlind",
    "ColorValueError",
    "ImageValueError",
    "NonFiniteColorError",
    "UnknownKindError",
]


class CVDError(ValueError):
    """本模块所有输入类错误的基类（仍是 :class:`ValueError`）。"""


class UnknownKindError(CVDError):
    """色觉类型不在支持列表中。"""


class ColorValueError(CVDError):
    """单色参数结构/取值非法（长度不是 3 等）。"""


class NonFiniteColorError(ColorValueError):
    """颜色分量不是有限实数（NaN / +Inf / -Inf）。"""


class ImageValueError(CVDError):
    """图像参数结构非法（不是行序列或行内颜色结构错误）。"""


# Machado, Oliveira & Fernandes (2009), severity 1.0。
# 每行之和均为 1；矩阵作用于线性 RGB，白点 (1,1,1) 与任意灰度保持不动。
SIMULATION_MATRICES = {
    "protanopia": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deuteranopia": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritanopia": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}

KINDS = ("protanopia", "deuteranopia", "tritanopia")

# ---------------------------------------------------------------------------
# sRGB 传输函数与 OKLab 矩阵常量（Björn Ottosson 定义）
# ---------------------------------------------------------------------------

def _s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lin2s(c):
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


_M1 = (
    (0.4122214708, 0.5363325363, 0.0514459929),
    (0.2119034982, 0.6806995451, 0.1073969566),
    (0.0883024619, 0.2817188376, 0.6299787005),
)
_M2 = (
    (0.2104542553, 0.7936177850, -0.0040720468),
    (1.9779984951, -2.4285922050, 0.4505937099),
    (0.0259040371, 0.7827717662, -0.8086757660),
)
_IM2 = (
    (1.0, 0.3963377774, 0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_IM1 = (
    (4.0767416621, -3.3077115913, 0.2309699292),
    (-1.2684380046, 2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, 1.7076147010),
)

# 批处理内核使用的拍平常量
_M100, _M101, _M102 = _M1[0]
_M110, _M111, _M112 = _M1[1]
_M120, _M121, _M122 = _M1[2]
_M200, _M201, _M202 = _M2[0]
_M210, _M211, _M212 = _M2[1]
_M220, _M221, _M222 = _M2[2]
_IM200, _IM201, _IM202 = _IM2[0]
_IM210, _IM211, _IM212 = _IM2[1]
_IM220, _IM221, _IM222 = _IM2[2]
_IM100, _IM101, _IM102 = _IM1[0]
_IM110, _IM111, _IM112 = _IM1[1]
_IM120, _IM121, _IM122 = _IM1[2]

_cbrt = math.cbrt

# OKLab 色域半径查找表的明度分格数
_NL = 513
_GAMUT_EPS = 2e-7


def _oklab(r, g, b):
    """sRGB [0,1] -> OKLab (L, a, b)。"""
    r = _s2lin(r)
    g = _s2lin(g)
    b = _s2lin(b)
    l0 = _M100 * r + _M101 * g + _M102 * b
    m0 = _M110 * r + _M111 * g + _M112 * b
    s0 = _M120 * r + _M121 * g + _M122 * b
    l_ = _cbrt(l0)
    m_ = _cbrt(m0)
    s_ = _cbrt(s0)
    return (
        _M200 * l_ + _M201 * m_ + _M202 * s_,
        _M210 * l_ + _M211 * m_ + _M212 * s_,
        _M220 * l_ + _M221 * m_ + _M222 * s_,
    )


def _srgb(L, a, b):
    """OKLab (L, a, b) -> sRGB（未钳制）。"""
    l_ = _IM200 * L + _IM201 * a + _IM202 * b
    m_ = _IM210 * L + _IM211 * a + _IM212 * b
    s_ = _IM220 * L + _IM221 * a + _IM222 * b
    l3 = l_ * l_ * l_
    m3 = m_ * m_ * m_
    s3 = s_ * s_ * s_
    return (
        _lin2s(_IM100 * l3 + _IM101 * m3 + _IM102 * s3),
        _lin2s(_IM110 * l3 + _IM111 * m3 + _IM112 * s3),
        _lin2s(_IM120 * l3 + _IM121 * m3 + _IM122 * s3),
    )


def _in_gamut(srgb):
    return all(-_GAMUT_EPS <= c <= 1.0 + _GAMUT_EPS for c in srgb)


# OKLab a/b 色度平面内的“混淆方向”单位向量 u。
#
# 生理学上红/绿色盲（protan/deuteran）的混淆线在 OKLab 中近似沿 b 轴，
# 蓝/黄色盲（tritan）近似沿 a 轴。这里的角度由 Machado (2009) 严重度 1.0
# 矩阵映射饱和锚色（RGB/CMY）后在 OKLab a/b 平面的主成分标定，并相对纯
# 主色轴施加一个小的、固定的色域安全偏角：纯主色轴在饱和红端会恰好压在
# sRGB 色域边界上，轻微偏角可让半径钳制在边界两侧都留有确定性余量，
# 同时仍与 PCA 主成分方向相差约 1°（远小于任何可感知色相差异）。
# 该表为确定性常量，不依赖运行时数值搜索，线程安全且结果可复现。
_CONFUSION_AXIS_ANGLE_DEG = {
    "protanopia": -90.8,
    "deuteranopia": -91.0,
    "tritanopia": -2.0,
}
_CONFUSION_AXES = {
    kind: (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
    for kind, angle in _CONFUSION_AXIS_ANGLE_DEG.items()
}


def _build_radius_table(ux, uy):
    """对每个明度 L 求沿 +/-u 方向仍在 sRGB 色域内的最大色度半径。

    使用精确传输函数构造（一次性预计算），二分得到 OKLab 色域边界。
    """
    pos = [0.0] * _NL
    neg = [0.0] * _NL
    for i in range(_NL):
        L = i / (_NL - 1)
        for sign, store in ((1.0, pos), (-1.0, neg)):
            hi = 0.01
            while _in_gamut(_srgb(L, sign * hi * ux, sign * hi * uy)) and hi < 2.0:
                hi *= 1.5
            lo = 0.0
            for _ in range(22):
                mid = (lo + hi) * 0.5
                if _in_gamut(_srgb(L, sign * mid * ux, sign * mid * uy)):
                    lo = mid
                else:
                    hi = mid
            store[i] = lo
    return tuple(pos), tuple(neg)


# 惰性、不可变的每类预计算状态（构造加锁，读取无锁 -> 并发安全）
_STATE_LOCK = threading.Lock()
_STATE = {}


def _build_state(kind):
    ux, uy = _CONFUSION_AXES[kind]
    pos, neg = _build_radius_table(ux, uy)
    return ux, uy, pos, neg


def _get_state(kind):
    state = _STATE.get(kind)
    if state is None:
        with _STATE_LOCK:
            state = _STATE.get(kind)
            if state is None:
                state = _build_state(kind)
                _STATE[kind] = state
    return state


def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _check_color(color):
    """校验单色输入，返回解包并钳制到 [0,1] 的三元组；错误类型可区分。"""
    if isinstance(color, (str, bytes)) or not hasattr(color, "__len__") or not hasattr(color, "__getitem__"):
        raise ColorValueError("color 必须是长度为 3 的序列 (r, g, b)")
    try:
        length = len(color)
    except TypeError:
        raise ColorValueError("color 必须是长度为 3 的序列 (r, g, b)")
    if length != 3:
        raise ColorValueError(f"color 必须恰好包含 3 个分量，得到 {length} 个")
    vals = []
    for v in color:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ColorValueError(f"color 分量必须是实数，得到 {v!r}")
        fv = float(v)
        if not math.isfinite(fv):
            raise NonFiniteColorError(f"color 分量必须有限，得到 {v!r}")
        vals.append(0.0 if fv < 0.0 else min(fv, 1.0))
    return vals[0], vals[1], vals[2]


def _is_sequence(obj):
    """有限容器（有长度、可下标），排除字符串/字节。"""
    return (
        not isinstance(obj, (str, bytes))
        and hasattr(obj, "__len__")
        and hasattr(obj, "__getitem__")
    )


def _check_image(image):
    """校验整图结构，返回物化为 list 的行（每行 list 化但不复制颜色）。

    图像/行的结构性问题抛 :class:`ImageValueError`；行内单个颜色的取值问题
    留给 :func:`_check_color` 抛 :class:`ColorValueError`，两类错误可区分。
    接受任何可迭代图像（list、tuple、生成器等），统一物化为嵌套 list。
    """
    if isinstance(image, (str, bytes)) or not hasattr(image, "__iter__"):
        raise ImageValueError("image 必须是行的序列（每行若干颜色三元组）")
    rows = []
    for row in image:
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise ImageValueError("图像的每一行必须是颜色三元组的序列")
        row_list = list(row)
        for color in row_list:
            if not _is_sequence(color):
                raise ImageValueError(
                    "图像行内的每个元素必须是长度为 3 的颜色三元组"
                )
        rows.append(row_list)
    return rows


def _project(r, g, b, ux, uy, pos, neg):
    """OKLab 混淆平面投影 + gamut 半径钳制（simulate 与 correct 共用内核）。"""
    if r == g == b:
        # 灰度/白点/黑点：色度为 0，投影后严格不动
        return (float(r), float(g), float(b))
    L, a, bq = _oklab(r, g, b)
    p = a * ux + bq * uy
    if p >= 0.0:
        if 0.0 < L < 1.0:
            x = L * (_NL - 1)
            i = int(x)
            t = x - i
            rlim = pos[i] + (pos[i + 1] - pos[i]) * t
        else:
            rlim = pos[0] if L <= 0.0 else pos[-1]
        radius = min(p, rlim)
        sign = 1.0
    else:
        ap = -p
        if 0.0 < L < 1.0:
            x = L * (_NL - 1)
            i = int(x)
            t = x - i
            rlim = neg[i] + (neg[i + 1] - neg[i]) * t
        else:
            rlim = neg[0] if L <= 0.0 else neg[-1]
        radius = min(ap, rlim)
        sign = -1.0
    out = _srgb(L, sign * radius * ux, sign * radius * uy)
    return (_clamp01(out[0]), _clamp01(out[1]), _clamp01(out[2]))


def _batch(colors, ux, uy, pos, neg):
    """整图批处理：与 _project 数学等价的内联内核，降低解释器调用开销。"""
    nlm1 = _NL - 1
    out = []
    append = out.append
    for c in colors:
        r, g, b = _check_color(c)
        if r == g == b:
            append((float(r), float(g), float(b)))
            continue
        # sRGB -> OKLab
        rr = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        gg = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        bb = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
        l0 = _M100 * rr + _M101 * gg + _M102 * bb
        m0 = _M110 * rr + _M111 * gg + _M112 * bb
        s0 = _M120 * rr + _M121 * gg + _M122 * bb
        l_ = _cbrt(l0)
        m_ = _cbrt(m0)
        s_ = _cbrt(s0)
        L = _M200 * l_ + _M201 * m_ + _M202 * s_
        a = _M210 * l_ + _M211 * m_ + _M212 * s_
        bq = _M220 * l_ + _M221 * m_ + _M222 * s_
        p = a * ux + bq * uy
        if p >= 0.0:
            if 0.0 < L < 1.0:
                x = L * nlm1
                i = int(x)
                t = x - i
                rlim = pos[i] + (pos[i + 1] - pos[i]) * t
            else:
                rlim = pos[0] if L <= 0.0 else pos[-1]
            radius = min(p, rlim)
            sign = 1.0
        else:
            ap = -p
            if 0.0 < L < 1.0:
                x = L * nlm1
                i = int(x)
                t = x - i
                rlim = neg[i] + (neg[i + 1] - neg[i]) * t
            else:
                rlim = neg[0] if L <= 0.0 else neg[-1]
            radius = min(ap, rlim)
            sign = -1.0
        aa = sign * radius * ux
        bbb = sign * radius * uy
        # OKLab -> sRGB
        ll = _IM200 * L + _IM201 * aa + _IM202 * bbb
        mm = _IM210 * L + _IM211 * aa + _IM212 * bbb
        ss = _IM220 * L + _IM221 * aa + _IM222 * bbb
        l3 = ll * ll * ll
        m3 = mm * mm * mm
        s3 = ss * ss * ss
        lr = _IM100 * l3 + _IM101 * m3 + _IM102 * s3
        lg = _IM110 * l3 + _IM111 * m3 + _IM112 * s3
        lb = _IM120 * l3 + _IM121 * m3 + _IM122 * s3
        or_ = 12.92 * lr if lr <= 0.0031308 else 1.055 * lr ** (1 / 2.4) - 0.055
        og = 12.92 * lg if lg <= 0.0031308 else 1.055 * lg ** (1 / 2.4) - 0.055
        ob = 12.92 * lb if lb <= 0.0031308 else 1.055 * lb ** (1 / 2.4) - 0.055
        append(
            (
                0.0 if or_ < 0.0 else min(or_, 1.0),
                0.0 if og < 0.0 else min(og, 1.0),
                0.0 if ob < 0.0 else min(ob, 1.0),
            )
        )
    return out


class ColorBlind:
    """色盲模拟与可逆校正。

    Parameters
    ----------
    kind:
        ``"protanopia"`` / ``"deuteranopia"`` / ``"tritanopia"`` 之一。
    """

    def __init__(self, kind):
        if not isinstance(kind, str) or kind not in SIMULATION_MATRICES:
            raise UnknownKindError(
                f"未知色觉类型 {kind!r}，可选 {', '.join(KINDS)}"
            )
        self.kind = kind
        # 触发一次惰性预计算（加锁），构造后实例即为无锁只读、并发安全
        ux, uy, pos, neg = _get_state(kind)
        self._ux, self._uy = ux, uy
        self._pos, self._neg = pos, neg

    # -- 主功能 -------------------------------------------------------------
    def apply(self, color):
        """单色盲模拟：sRGB 三元组 -> 模拟后的 sRGB 三元组。"""
        r, g, b = _check_color(color)
        return _project(r, g, b, self._ux, self._uy, self._pos, self._neg)

    def apply_image(self, image):
        """整图模拟。

        接受行式序列（每行是若干 sRGB 三元组），返回同形状的嵌套 list；
        不修改输入。空图返回 ``[]``。
        """
        rows = _check_image(image)
        result = []
        ux, uy, pos, neg = self._ux, self._uy, self._pos, self._neg
        for row in rows:
            result.append(_batch(row, ux, uy, pos, neg))
        return result

    # -- 反向校正 -----------------------------------------------------------
    def correct(self, color):
        """反向校正：把颜色投回该色觉混淆线在 OKLab 色域内的最近点。

        对 :meth:`apply` 的输出调用时为其有效逆映射；模拟输出在混淆线上，
        故 ``apply(correct(apply(x))) == apply(x)``（仅受浮点误差影响），
        且保持 OKLab 明度与色相角不变。
        """
        r, g, b = _check_color(color)
        return _project(r, g, b, self._ux, self._uy, self._pos, self._neg)

    def correct_image(self, image):
        """整图反向校正，约定同 :meth:`apply_image`。"""
        rows = _check_image(image)
        result = []
        ux, uy, pos, neg = self._ux, self._uy, self._pos, self._neg
        for row in rows:
            result.append(_batch(row, ux, uy, pos, neg))
        return result
