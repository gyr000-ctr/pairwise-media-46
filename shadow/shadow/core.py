"""2D 投影阴影多边形生成。

光源是平面 ``z = 0`` 上方的一点 ``(lx, ly, lz)``。多边形的每个顶点沿
光源射线投影到物体平面下方 1 个单位处的平行承接平面上，等价于以光源
的地面位置为中心，按 ``(lz + 1) / lz`` 做位似缩放。因此：

* 缩放系数大于 1，阴影多边形包含全部被投影顶点；
* 面积按该系数的平方变化，随光源高度 ``lz`` 严格单调递减；
* 光源与物体共面（``lz == 0``）时无法投影：若其地面位置落在物体内部，
  报 ``ValueError``（光源在物体内），否则同样因投影发散而报错；
* 连续重合的退化边只保留一个顶点；
* 投影状态只在实例上保存，并加锁更新，可被多线程安全调用。
"""

import math
import threading

__all__ = ["Shadow"]

# 阴影承接平面位于物体平面下方 1 个单位处。
_RECEIVER_DEPTH = 1.0
# 判断光源是否落在多边形边界上时使用的相对容差。
_EDGE_EPS = 1e-12


def _as_float(value, what):
    """把一个输入坐标严格转换为有限 float，并分类报错。"""
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise TypeError("%s must be a real number, got %s"
                        % (what, type(value).__name__))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise TypeError("%s must be a real number, got %s"
                        % (what, type(value).__name__))
    if not math.isfinite(result):
        raise ValueError("%s must be finite, got %r" % (what, value))
    return result


def _as_sequence(value, what):
    if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
        raise TypeError("%s must be an iterable" % (what,))
    try:
        return list(value)
    except TypeError:
        raise TypeError("%s must be an iterable" % (what,))


def _parse_light(light):
    values = _as_sequence(light, "light")
    if len(values) != 3:
        raise ValueError("light must contain exactly three coordinates")
    lx = _as_float(values[0], "light x")
    ly = _as_float(values[1], "light y")
    lz = _as_float(values[2], "light z")
    if lz < 0.0:
        raise ValueError("light height must be non-negative, got %r" % (lz,))
    return lx, ly, lz


def _parse_points(points):
    raw = _as_sequence(points, "points")
    if not raw:
        raise ValueError("points must not be empty")
    parsed = []
    for index, point in enumerate(raw):
        coords = _as_sequence(point, "points[%d]" % (index,))
        if len(coords) != 2:
            raise ValueError("points[%d] must contain exactly two coordinates"
                             % (index,))
        parsed.append((_as_float(coords[0], "points[%d].x" % (index,)),
                       _as_float(coords[1], "points[%d].y" % (index,))))
    return parsed


def _drop_degenerate_vertices(points):
    """删除连续重复顶点以及闭合处重复的起点。"""
    if not points:
        return []
    result = [points[0]]
    for point in points[1:]:
        if point != result[-1]:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def _polygon_area(points):
    """鞋带公式，退化多边形（少于 3 个顶点）面积为 0。

    先按坐标量级归一化再求和，避免大坐标直接相乘溢出。
    """
    count = len(points)
    if count < 3:
        return 0.0
    magnitude = max(max(abs(x), abs(y)) for x, y in points)
    if magnitude == 0.0:
        return 0.0
    terms = []
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        terms.append((x1 / magnitude) * (y2 / magnitude)
                     - (x2 / magnitude) * (y1 / magnitude))
    return abs(math.fsum(terms)) * 0.5 * magnitude * magnitude


def _point_on_segment(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return px == ax and py == ay
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    if t < -_EDGE_EPS or t > 1.0 + _EDGE_EPS:
        return False
    cx = ax + t * dx
    cy = ay + t * dy
    tolerance = _EDGE_EPS * (1.0 + abs(ax) + abs(ay) + abs(bx) + abs(by))
    return (px - cx) ** 2 + (py - cy) ** 2 <= tolerance * tolerance


def _point_in_polygon(px, py, points):
    """射线法判断点是否在多边形内部或边界上。"""
    count = len(points)
    if count == 0:
        return False
    edge_count = count if count > 1 else 1
    for index in range(edge_count):
        ax, ay = points[index]
        bx, by = points[(index + 1) % count] if count > 1 else points[index]
        if _point_on_segment(px, py, ax, ay, bx, by):
            return True
    if count < 3:
        return False
    inside = False
    previous = count - 1
    for index in range(count):
        xi, yi = points[index]
        xj, yj = points[previous]
        if (yi > py) != (yj > py):
            crossing = (xj - xi) * (py - yi) / (yj - yi) + xi
            if px < crossing:
                inside = not inside
        previous = index
    return inside


class Shadow:
    """由一个点光源把 2D 多边形投影为阴影多边形。"""

    def __init__(self, light):
        """按光源 ``(x, y, z)`` 构造。

        非数值输入报 :class:`TypeError`；非有限值或负高度报
        :class:`ValueError`。``z == 0`` 在构造时被接受，投影时再判定光源
        是否位于物体内。
        """
        self._light = _parse_light(light)
        self._lock = threading.RLock()
        self._shadow = None
        self._area = None

    def project(self, points):
        """生成阴影多边形，返回 ``[(x, y), ...]``（float 坐标）。"""
        parsed = _parse_points(points)
        polygon = _drop_degenerate_vertices(parsed)
        lx, ly, lz = self._light

        if lz == 0.0:
            if _point_in_polygon(lx, ly, polygon):
                raise ValueError("light source lies inside the polygon")
            raise ValueError("light source lies in the polygon plane; "
                             "shadow is undefined")

        scale = (lz + _RECEIVER_DEPTH) / lz
        if not math.isfinite(scale):
            raise ValueError("light source is too close to the polygon plane")

        result = [(lx + (x - lx) * scale, ly + (y - ly) * scale)
                  for x, y in polygon]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in result):
            raise ValueError("projected coordinates are not finite")
        # 位似变换下面积 = 原面积 * scale^2，避免大坐标相乘溢出。
        area = _polygon_area(polygon) * scale * scale
        if not math.isfinite(area):
            raise ValueError("shadow area is not finite")

        with self._lock:
            self._shadow = tuple(result)
            self._area = area
        return list(result)

    def area(self):
        """返回最近一次 :meth:`project` 生成的阴影面积。"""
        with self._lock:
            if self._area is None:
                raise ValueError("project() must be called before area()")
            return self._area
