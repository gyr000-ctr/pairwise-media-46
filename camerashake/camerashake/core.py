"""camerashake 核心 API。

相机抖动：叠加多个正弦分量，能量集中在 :attr:`CameraShake.BAND_HZ` 频段；
强度随时间平滑衰减到 0；同一 seed 在相同 ``dt`` 序列下逐位可复现。
偏移模长任何时刻都不超过设定强度。
"""

import math
import numbers
import random
import threading

__all__ = ["CameraShake"]

_TAU = 2.0 * math.pi


def _is_real(value):
    """实数且不是 bool / 非数字。"""
    if isinstance(value, bool):
        return False
    return isinstance(value, numbers.Real)


def _finite_real(name, value):
    """校验 *name* 为有限实数。"""
    if not _is_real(value):
        raise TypeError("%s 必须是实数，收到 %s" % (name, type(value).__name__))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s 必须是有限值，收到 %r" % (name, value))
    return value


def _seed_value(seed):
    """seed 必须是整数；浮点/非数字给 TypeError，非有限值给 ValueError。"""
    if isinstance(seed, bool):
        raise TypeError("seed 必须是整数，收到 bool")
    if isinstance(seed, numbers.Integral):
        return int(seed)
    if isinstance(seed, numbers.Real):
        value = float(seed)
        if not math.isfinite(value):
            raise ValueError("seed 必须是有限值，收到 %r" % (value,))
        raise TypeError("seed 必须是整数，收到 float")
    raise TypeError("seed 必须是整数，收到 %s" % (type(seed).__name__,))


def _delta_time(dt):
    """校验推进步长：实数、有限、非负。"""
    if not _is_real(dt):
        raise TypeError("dt 必须是实数，收到 %s" % (type(dt).__name__,))
    dt = float(dt)
    if not math.isfinite(dt):
        raise ValueError("dt 必须是有限值，收到 %r" % (dt,))
    if dt < 0.0:
        raise ValueError("dt 不能为负数，收到 %r" % (dt,))
    return dt


class CameraShake:
    """确定性的二维相机抖动。

    Parameters
    ----------
    strength:
        最大位移（> 0，有限）。任意轴分量与偏移模长都不超过该值。
    duration:
        持续时间（秒，> 0，有限）。到期后 ``remaining()`` 恒为 0，
        ``update()`` 恒返回 ``(0.0, 0.0)``。
    seed:
        整数种子；同一 seed 生成的频率/相位完全一致。

    Notes
    -----
    抖动由若干正弦分量叠加而成，各分量频率按 seed 分层分布在
    :attr:`BAND_HZ` 内，因此频谱能量集中于该频段。包络在开头有极短的
    淡入、随后平滑衰减到 0，保证 t=0 与结束时刻输出均为精确的 0。
    实例状态只在 ``update``/``remaining`` 内由锁保护更新，无全局状态。
    """

    #: 抖动能量集中的频段（Hz）。
    BAND_HZ = (2.0, 8.0)
    #: 叠加的正弦分量个数。
    COMPONENT_COUNT = 5

    __slots__ = (
        "_strength",
        "_duration",
        "_seed",
        "_inv_n",
        "_omega",
        "_phase",
        "_attack",
        "_t",
        "_ended",
        "_end_eps",
        "_lock",
    )

    def __init__(self, strength, duration, seed=0):
        strength = _finite_real("strength", strength)
        duration = _finite_real("duration", duration)
        if strength <= 0.0:
            raise ValueError("strength 必须大于 0，收到 %r" % (strength,))
        if duration <= 0.0:
            raise ValueError("duration 必须大于 0，收到 %r" % (duration,))
        seed = _seed_value(seed)

        n = int(self.COMPONENT_COUNT)
        if n < 2:  # 至少两个分量才能构成多频率叠加。
            n = 2
        lo, hi = self.BAND_HZ

        rnd = random.Random(seed)
        omega = []
        phase = []
        for i in range(n):
            # 分层抖动：分量分散在频段内且互不重合。
            frac = (i + 0.15 + 0.7 * rnd.random()) / n
            omega.append(_TAU * (lo + (hi - lo) * frac))
            phase.append(_TAU * rnd.random())

        self._strength = strength
        self._duration = duration
        self._seed = seed
        self._inv_n = 1.0 / n
        self._omega = tuple(omega)
        self._phase = tuple(phase)
        # 起始极短淡入，使 t=0 精确为 (0, 0) 且无跳变。
        self._attack = min(0.02, duration * 0.5)
        self._t = 0.0
        self._ended = False
        # 吸收 duration/dt 整数步累加的舍入误差，使收尾不晚于 duration。
        self._end_eps = duration * 1e-12
        self._lock = threading.Lock()

    @property
    def strength(self):
        """设定的最大位移。"""
        return self._strength

    @property
    def duration(self):
        """设定的持续时间（秒）。"""
        return self._duration

    @property
    def seed(self):
        """构造时使用的整数种子。"""
        return self._seed

    def update(self, dt):
        """推进 *dt* 秒，返回本帧平移偏移 ``(dx, dy)``。

        ``dt`` 必须是非负有限实数；越界或类型错误分别抛
        :class:`ValueError` / :class:`TypeError`。
        """
        dt = _delta_time(dt)
        lock = self._lock
        with lock:
            if self._ended:
                return (0.0, 0.0)

            duration = self._duration
            t = self._t + dt
            if t >= duration - self._end_eps:
                self._t = duration
                self._ended = True
                return (0.0, 0.0)

            self._t = t
            if t <= 0.0:
                return (0.0, 0.0)

            u = t / duration
            envelope = (1.0 - u) * (1.0 - u)
            attack = self._attack
            if t < attack:
                q = t / attack
                envelope *= q * q * (3.0 - 2.0 * q)

            sx = 0.0
            sy = 0.0
            for omega, phase in zip(self._omega, self._phase):
                angle = omega * t + phase
                sx += math.cos(angle)
                sy += math.sin(angle)

            # 每分量幅度 1/n，故 |(sx, sy)| <= 1，乘以 strength 后不超过强度。
            scale = self._strength * envelope * self._inv_n
            return (sx * scale, sy * scale)

    def remaining(self):
        """剩余强度：从 ``strength`` 单调不增，到期精确为 ``0.0``。"""
        with self._lock:
            if self._ended:
                return 0.0
            u = self._t / self._duration
            if u >= 1.0:
                return 0.0
            d = 1.0 - u
            return self._strength * d * d
