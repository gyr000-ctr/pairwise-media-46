"""hueshift 核心 API：在 OKLCH 感知空间中旋转色相。

设计要点：
- 色相旋转在 OKLab 的 a-b 平面内完成（等价于 OKLCH 中保持 L 与 C 不变、
  仅平移 H），避免逐像素 atan2，直接复用预计算的 cos/sin。
- 0 度（含 360 度取模后）走恒等快路径，原样返回。
- 灰度（r == g == b）色度为 0，旋转不变，走快路径。
- 全部状态在构造时预计算，apply 只读共享状态 + 幂等缓存，并发安全。
"""

import math

__all__ = ["HueShift"]

_CACHE_LIMIT = 262144


class HueShift:
    """按给定角度（度）在 OKLCH 空间旋转 sRGB 颜色的色相。"""

    __slots__ = ("_cos", "_sin", "_identity", "_cache")

    def __init__(self, degrees):
        """构造。degrees 必须在 [0, 360] 内且为有限实数，否则报错。"""
        if isinstance(degrees, bool) or not isinstance(degrees, (int, float)):
            raise TypeError("degrees must be a real number")
        d = float(degrees)
        if not math.isfinite(d):
            raise ValueError("degrees must be finite")
        if d < 0.0 or d > 360.0:
            raise ValueError("degrees out of range [0, 360]")
        d = d % 360.0  # 360 -> 0，保证 360 度精确回到原色
        rad = math.radians(d)
        self._cos = math.cos(rad)
        self._sin = math.sin(rad)
        self._identity = d == 0.0
        self._cache = {}

    def apply(self, color):
        """转换单色。输入/输出均为 (r, g, b) 浮点三元组，分量在 [0, 1]。"""
        try:
            r, g, b = color
        except (TypeError, ValueError):
            raise ValueError("color must be a 3-component iterable") from None
        r = float(r)
        g = float(g)
        b = float(b)
        if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
            raise ValueError("color components must be finite")
        if not (0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0):
            raise ValueError("color components out of range [0, 1]")

        if self._identity or (r == g == b):
            return (r, g, b)

        key = (r, g, b)
        cache = self._cache
        hit = cache.get(key)
        if hit is not None:
            return hit

        # sRGB → 线性
        rl = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
        gl = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
        bl = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4

        # 线性 sRGB → LMS → OKLab
        l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl
        m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl
        s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl
        l_ = math.cbrt(l)
        m_ = math.cbrt(m)
        s_ = math.cbrt(s)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        # OKLCH 色相旋转：L 与 C=sqrt(a²+b²) 不变，仅 H 平移
        cos = self._cos
        sin = self._sin
        a2 = a * cos - bb * sin
        b2 = a * sin + bb * cos

        # OKLab → LMS → 线性 sRGB
        l_ = L + 0.3963377774 * a2 + 0.2158037573 * b2
        m_ = L - 0.1055613458 * a2 - 0.0638541728 * b2
        s_ = L - 0.0894841775 * a2 - 1.2914855480 * b2
        l = l_ * l_ * l_
        m = m_ * m_ * m_
        s = s_ * s_ * s_
        rl = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

        # 线性 → sRGB（先做色域裁剪，越界颜色截断到 [0, 1]）
        out = (_encode(rl), _encode(gl), _encode(bl))
        if len(cache) < _CACHE_LIMIT:
            cache[key] = out
        return out


def _encode(c):
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055
