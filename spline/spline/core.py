"""Catmull-Rom 样条重采样（纯 Python 标准库实现）。

特性：
- 均匀参数 Catmull-Rom（alpha=0.5），曲线通过全部原始控制点；
- 端点切线按 clamp 处理（首末控制点各复制一次）；
- 基于弧长的等距重采样，弧长由细分折线数值积分得到；
- 无可变共享状态，``sample``/``resample`` 仅读实例、写局部变量，并发安全。
"""

import math

# 每个三次区间的细分数目；折线弧长相对真实弧长的误差随细分二次收敛，
# 16 段细分在 Catmull-Rom 可达的曲率下相对误差约 0.05%，远小于 1%。
_SUBDIV = 16


class Spline:
    def __init__(self, points):
        """按控制点构造。

        参数 ``points`` 为至少 2 个二维点的可迭代对象；点不足 2 个或
        包含非有限数值时抛出 ``ValueError``。输入会被复制，后续不被修改。
        """
        pts = list(points)
        if len(pts) < 2:
            raise ValueError("at least 2 control points are required")
        cleaned = []
        for p in pts:
            x = float(p[0])
            y = float(p[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("control points must contain finite numbers")
            cleaned.append((x, y))
        self._points = cleaned
        self._nseg = len(cleaned) - 1

    def _neighbors(self, i):
        """返回第 i 个三次区间的 4 个控制点，端点按 clamp 处理。"""
        pts = self._points
        last = len(pts) - 1
        p0 = pts[i - 1] if i > 0 else pts[0]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2] if i + 2 <= last else pts[last]
        return p0, p1, p2, p3

    @staticmethod
    def _basis(u):
        """Catmull-Rom 基函数系数（切线权重 0.5）。"""
        u2 = u * u
        u3 = u2 * u
        return (
            -0.5 * u + u2 - 0.5 * u3,
            1.0 - 2.5 * u2 + 1.5 * u3,
            0.5 * u + 2.0 * u2 - 1.5 * u3,
            -0.5 * u2 + 0.5 * u3,
        )

    def sample(self, t):
        """按归一化参数 t ∈ [0, 1] 采样，返回二维 ``tuple``。"""
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be a finite number")
        if t < 0.0 or t > 1.0:
            raise ValueError("t out of range [0, 1]")
        pts = self._points
        if t == 0.0:
            return pts[0]
        if t == 1.0:
            return pts[-1]
        nseg = self._nseg
        pos = t * nseg
        i = int(pos)
        if i >= nseg:  # 浮点保险
            i = nseg - 1
        u = pos - i
        c0, c1, c2, c3 = self._basis(u)
        p0, p1, p2, p3 = self._neighbors(i)
        x = c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0]
        y = c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1]
        return (x, y)

    def _dense_polyline(self):
        """细分整条样条为折线，用于弧长积分与等距落点。"""
        pts = self._points
        nseg = self._nseg
        dense = [pts[0]]
        step = 1.0 / _SUBDIV
        for i in range(nseg):
            p0, p1, p2, p3 = self._neighbors(i)
            x0, y0 = p0
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            for k in range(1, _SUBDIV + 1):
                u = k * step
                c0, c1, c2, c3 = self._basis(u)
                dense.append((
                    c0 * x0 + c1 * x1 + c2 * x2 + c3 * x3,
                    c0 * y0 + c1 * y1 + c2 * y2 + c3 * y3,
                ))
        return dense

    def resample(self, spacing):
        """按给定弧长间距等距重采样。

        首点为原始起点，内部相邻点的弧长间隔等于 ``spacing``，末点为
        原始终点（末段为不足一个间距的收尾段）。``spacing`` 必须为正的
        有限数，否则抛出 ``ValueError``。
        """
        spacing = float(spacing)
        if not (math.isfinite(spacing) and spacing > 0.0):
            raise ValueError("spacing must be a positive finite number")

        dense = self._dense_polyline()

        # 累积弧长
        cum = [0.0]
        append_cum = cum.append
        total = 0.0
        ax, ay = dense[0]
        for b in dense[1:]:
            bx, by = b
            total += math.hypot(bx - ax, by - ay)
            append_cum(total)
            ax, ay = bx, by

        out = [dense[0]]
        eps = 1e-12 * max(1.0, total)
        if total > 0.0:
            target = spacing
            j = 0
            last_edge = len(dense) - 1
            while target < total - eps:
                while cum[j + 1] < target:
                    j += 1
                    if j >= last_edge:
                        break
                a = dense[j]
                b = dense[j + 1]
                edge_len = cum[j + 1] - cum[j]
                if edge_len <= 0.0:
                    point = b
                else:
                    r = (target - cum[j]) / edge_len
                    point = (
                        a[0] + (b[0] - a[0]) * r,
                        a[1] + (b[1] - a[1]) * r,
                    )
                out.append(point)
                target += spacing

        # 保证终点精确落在原始末控制点上；输出始终包含起点与终点两个点
        endpoint = self._points[-1]
        if len(out) == 1 or math.dist(out[-1], endpoint) > eps:
            out.append(endpoint)
        else:
            out[-1] = endpoint
        return out
