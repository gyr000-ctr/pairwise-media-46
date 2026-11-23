"""audioonset 核心 API：多频带起音检测。

同时计算能量包络与谱通量两种特征并融合判定，支持迟滞
（高/低双阈值）、最小间隔、跨缓冲区状态连续。非法构造
参数抛出 ValueError；样本含 NaN/无穷抛出
NonFiniteSampleError（ValueError 子类，可区分）。
"""

from __future__ import annotations

import math
import threading

try:  # numpy 为可选加速后端，缺失时回退纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

__all__ = ["OnsetDetector", "NonFiniteSampleError"]

_FRAME = 16  # 分析帧长（样本数）
_HOP = 8  # 帧移（样本数）
_NBINS = _FRAME // 2 + 1


class NonFiniteSampleError(ValueError):
    """输入样本包含 NaN 或无穷。

    继承 ValueError，但类型与构造参数错误可区分。
    """


def _check_param(name, value):
    """非法参数（非数值、非有限、非正）一律 ValueError。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"非法参数 {name}={value!r}: 必须为正的有限数值")
    return float(value)


# 纯 Python 回退用的 DFT 旋转因子表
if _np is None:  # pragma: no cover
    _COS = [
        [math.cos(2.0 * math.pi * k * n / _FRAME) for n in range(_FRAME)]
        for k in range(_NBINS)
    ]
    _SIN = [
        [math.sin(2.0 * math.pi * k * n / _FRAME) for n in range(_FRAME)]
        for k in range(_NBINS)
    ]


def _dft_mags(frame):  # pragma: no cover - 仅无 numpy 时使用
    mags = []
    for k in range(_NBINS):
        re = 0.0
        im = 0.0
        ck = _COS[k]
        sk = _SIN[k]
        for n in range(_FRAME):
            x = frame[n]
            re += x * ck[n]
            im -= x * sk[n]
        mags.append(math.hypot(re, im))
    return mags


class OnsetDetector:
    """多频带起音检测器（能量包络 + 谱通量融合判定）。

    - rate: 采样率（Hz），必须为正的有限数值
    - threshold: 触发阈值（迟滞上阈），必须为正的有限数值
    - min_gap: 相邻起音的最小间隔（样本数），必须为正的有限数值

    process() 可多次调用，内部状态（包络、上一帧谱、迟滞
    状态、样本偏移）跨缓冲区连续；返回值为全局样本索引列表。
    线程安全：并发 process 调用由内部锁串行化。
    """

    def __init__(self, rate, threshold, min_gap):
        self._rate = _check_param("rate", rate)
        self._threshold = _check_param("threshold", threshold)
        self._min_gap = _check_param("min_gap", min_gap)
        self._low = self._threshold * 0.5  # 迟滞下阈
        self._lock = threading.Lock()
        # 跨缓冲区状态
        self._offset = 0  # _carry[0] 对应的全局样本索引
        self._last_onset = -math.inf
        self._armed = True
        self._prev_energy = 0.0
        self._prev_spectrum = [0.0] * _NBINS
        self._carry = []

    # ---- 输入校验 -------------------------------------------------

    @staticmethod
    def _validate(samples):
        """返回有限数值序列；非数值 -> ValueError，非有限 -> NonFiniteSampleError。"""
        if _np is not None:
            try:
                arr = _np.asarray(samples, dtype=_np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"samples 必须为数值序列: {exc}") from exc
            if arr.ndim != 1:
                raise ValueError("samples 必须为一维数值序列")
            if not _np.isfinite(arr).all():
                raise NonFiniteSampleError("样本包含 NaN 或无穷")
            return arr
        try:  # pragma: no cover
            vals = [float(s) for s in samples]
        except (TypeError, ValueError) as exc:  # pragma: no cover
            raise ValueError(f"samples 必须为数值序列: {exc}") from exc
        for v in vals:  # pragma: no cover
            if not math.isfinite(v):
                raise NonFiniteSampleError("样本包含 NaN 或无穷")
        return vals  # pragma: no cover

    # ---- 特征计算 -------------------------------------------------

    def _scores(self, buf, n_blocks):
        """逐块计算融合得分：0.5*能量包络上升 + 0.5*谱通量。"""
        if _np is not None:
            view = _np.lib.stride_tricks.sliding_window_view(buf, _FRAME)
            frames = view[::_HOP]
            energies = _np.sqrt((frames * frames).sum(axis=1) / _FRAME)
            mags = _np.abs(_np.fft.rfft(frames, axis=1))
            prev_e = _np.empty(n_blocks)
            prev_e[0] = self._prev_energy
            prev_e[1:] = energies[:-1]
            e_rise = _np.maximum(0.0, energies - prev_e)
            prev_s = _np.empty((n_blocks, _NBINS))
            prev_s[0] = self._prev_spectrum
            prev_s[1:] = mags[:-1]
            flux = _np.maximum(0.0, mags - prev_s).sum(axis=1) / _FRAME
            self._prev_energy = float(energies[-1])
            self._prev_spectrum = mags[-1].tolist()
            return (0.5 * e_rise + 0.5 * flux).tolist()
        scores = []  # pragma: no cover
        prev_e = self._prev_energy  # pragma: no cover
        prev_s = self._prev_spectrum  # pragma: no cover
        for i in range(n_blocks):  # pragma: no cover
            frame = buf[i * _HOP : i * _HOP + _FRAME]
            e = math.sqrt(sum(x * x for x in frame) / _FRAME)
            mag = _dft_mags(frame)
            flux = 0.0
            for k in range(_NBINS):
                d = mag[k] - prev_s[k]
                if d > 0.0:
                    flux += d
            rise = e - prev_e
            scores.append(0.5 * (rise if rise > 0.0 else 0.0) + 0.5 * flux / _FRAME)
            prev_e = e
            prev_s = mag
        self._prev_energy = prev_e  # pragma: no cover
        self._prev_spectrum = prev_s  # pragma: no cover
        return scores  # pragma: no cover

    # ---- 主入口 ---------------------------------------------------

    def process(self, samples):
        """处理一段样本，返回起音位置列表（跨缓冲区连续的全局索引）。"""
        arr = self._validate(samples)
        with self._lock:
            if len(arr) == 0:
                return []
            if _np is not None:
                buf = (
                    _np.concatenate([_np.asarray(self._carry), arr])
                    if self._carry
                    else arr
                )
            else:  # pragma: no cover
                buf = self._carry + arr
            n = len(buf)
            if n < _FRAME:
                self._carry = buf.tolist() if _np is not None else buf
                return []
            n_blocks = (n - _FRAME) // _HOP + 1
            # 只消费到下一个块起点，余下样本留待下次（保证跨缓冲区块网格对齐）
            used = n_blocks * _HOP
            tail = buf[used:]
            self._carry = tail.tolist() if _np is not None else list(tail)

            scores = self._scores(buf, n_blocks)

            # 迟滞 + 最小间隔判定
            onsets = []
            armed = self._armed
            last = self._last_onset
            high = self._threshold
            low = self._low
            gap = self._min_gap
            base = self._offset
            pos = base
            for s in scores:
                if armed:
                    if s > high:
                        if pos - last >= gap:
                            onsets.append(pos)
                            last = pos
                        armed = False
                elif s < low:
                    armed = True
                pos += _HOP
            self._armed = armed
            self._last_onset = last
            self._offset = base + used
            return onsets
