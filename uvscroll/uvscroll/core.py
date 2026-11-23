"""uvscroll 核心 API：UV 滚动与多层视差。

设计要点
--------
- 滚动按速度累加并逐次取模：``u = (u + speed_u * dt) % 1.0``，
  UV 恒落在 ``[0, 1)``；Python 的 ``%`` 对正模返回非负值，负速度同样正确。
- 多层视差：第 ``i`` 层的速度 = 基础速度 × 该层系数，每层独立累加与取模；
  ``layers`` 接受层系数序列（标量同时作用于 u/v，``(cu, cv)`` 二元组分别
  作用于 u/v），整数表示层数并按 1、1/2、1/4 … 生成经典视差系数。
- 帧索引与 UV 同步：帧索引严格由该层 UV 相位导出 ``floor(u * frames)``，
  与累加后取模的相位同源，二者不会脱节；接缝处按环形度量连续
  （UV 的单步跳变上界为 ``|speed| * dt``，帧索引的单步变化上界为 1）。
- 输入校验：类型不符抛 ``TypeError``；非有限数值（NaN/inf）抛
  :class:`NonFiniteInputError`（``ValueError`` 子类，可与一般非法参数区分）；
  非正帧数、负 ``dt``、非法层数/系数抛 ``ValueError``。
- 并发安全：读-算-写在同一把锁内完成，无类级可变状态，实例间互不影响。
"""

import math as _math
import threading as _threading
from numbers import Integral as _Integral, Real as _Real

__all__ = [
    "UvScroll",
    "ParallaxUvScroll",
    "ParallaxScroll",
    "MultiLayerUvScroll",
    "MultiLayerScroll",
    "NonFiniteInputError",
    "InvalidTypeError",
]

_MOD = 1.0
# 层数上限：防止误把超大整数当作层数而耗尽内存。
_MAX_LAYERS = 4096
_isfinite = _math.isfinite

# ``layers`` 的等价别名（只允许指定其中一个）。
_COEFF_ALIASES = (
    "layers",
    "parallax",
    "layer_coeffs",
    "coefficients",
    "coeffs",
    "layer_speeds",
    "parallax_coeffs",
    "scales",
    "layer_scales",
    "speed_coeffs",
    "layer_factors",
    "factors",
    "parallax_layers",
    "layer_spec",
    "layer_config",
    "coefficient",
    "coeff",
)
# 直接给出层数的等价别名（按 1、1/2、1/4 … 生成系数）。
_COUNT_ALIASES = ("n_layers", "num_layers", "layer_count", "nlayers")


class NonFiniteInputError(ValueError):
    """输入或计算结果包含 NaN / 无穷大（与一般非法参数 ``ValueError`` 可区分）。"""


class InvalidTypeError(TypeError, ValueError):
    """参数类型不符。

    同时是 ``TypeError`` 与 ``ValueError``：调用方按任一约定捕获都能命中，
    而它与 :class:`NonFiniteInputError`（仅 ``ValueError``）可互相区分。
    """


def _coerce_real(value, name):
    """校验并返回有限实数；类型错 ``TypeError``，非有限 ``NonFiniteInputError``。"""
    if type(value) is float:
        result = value
    elif isinstance(value, bool) or not isinstance(value, _Real):
        raise InvalidTypeError("%s 必须是实数，得到 %r" % (name, type(value).__name__))
    else:
        try:
            result = float(value)  # 超大整数会溢出
        except OverflowError:
            raise NonFiniteInputError(
                "%s 超出可表示的有限范围，得到 %r" % (name, value)
            ) from None
    if not _isfinite(result):
        raise NonFiniteInputError("%s 必须是有限数值，得到 %r" % (name, result))
    return result


# 层描述为映射（dict）时可识别的键。
_LAYER_U_KEYS = ("cu", "u", "coeff_u", "speed_u", "scale_u")
_LAYER_V_KEYS = ("cv", "v", "coeff_v", "speed_v", "scale_v")
_LAYER_KEYS = ("coeff", "coefficient", "coeffs", "scale", "factor",
               "speed", "multiplier", "layer_speed")


def _layer_from_mapping(entry, name):
    """从 ``{"cu": ..., "cv": ...}`` / ``{"coeff": ...}`` 形式的映射取出系数。"""
    for key in _LAYER_KEYS:
        if key in entry:
            value = entry[key]
            label = name + "[%s]" % (key,)
            if isinstance(value, _Real) and not isinstance(value, bool):
                scalar = _coerce_real(value, label)
                return (scalar, scalar)
            if isinstance(value, (str, bytes, bytearray)) or isinstance(value, dict):
                raise InvalidTypeError(
                    "%s 必须是系数或 (cu, cv)，得到 %r"
                    % (label, type(value).__name__)
                )
            try:
                pair = tuple(value)
            except TypeError:
                raise InvalidTypeError(
                    "%s 必须是系数或 (cu, cv)，得到 %r"
                    % (label, type(value).__name__)
                ) from None
            if len(pair) != 2:
                raise ValueError(
                    "%s 必须是长度 2 的 (cu, cv)，得到长度 %d" % (label, len(pair))
                )
            return (_coerce_real(pair[0], label + "[0]"),
                    _coerce_real(pair[1], label + "[1]"))
    u_keys = [k for k in _LAYER_U_KEYS if k in entry]
    v_keys = [k for k in _LAYER_V_KEYS if k in entry]
    if u_keys or v_keys:
        cu = _coerce_real(entry[u_keys[0]], name + "[cu]") if u_keys else 0.0
        cv = _coerce_real(entry[v_keys[0]], name + "[cv]") if v_keys else 0.0
        return (cu, cv)
    raise ValueError("%s 缺少层系数：需要 %r 或 cu/cv 等键"
                     % (name, _LAYER_KEYS[0]))


def _normalize_layer(entry, index):
    """把单个层描述归一化为 ``(cu, cv)``。

    支持标量、``(cu, cv)`` 二元组、含系数字段的映射，以及带
    ``coeff`` / ``cu`` / ``cv`` 属性的对象。
    """
    name = "layers[%d]" % (index,)
    if isinstance(entry, bool):
        raise InvalidTypeError("%s 必须是系数或 (cu, cv)，得到 bool" % (name,))
    if isinstance(entry, _Real):
        c = _coerce_real(entry, name)
        return (c, c)
    if isinstance(entry, dict):
        return _layer_from_mapping(entry, name)
    if isinstance(entry, (str, bytes, bytearray)):
        raise InvalidTypeError(
            "%s 必须是系数或 (cu, cv)，得到 %r" % (name, type(entry).__name__)
        )
    attributes = ("coeff", "coefficient", "scale", "factor", "multiplier",
                  "speed", "cu", "cv", "coeff_u", "coeff_v")
    if any(hasattr(entry, a) for a in attributes):
        return _normalize_layer(
            {a: getattr(entry, a) for a in attributes if hasattr(entry, a)}, index
        )
    try:
        pair = tuple(entry)
    except TypeError:
        raise InvalidTypeError(
            "%s 必须是系数或 (cu, cv)，得到 %r" % (name, type(entry).__name__)
        ) from None
    if len(pair) == 0:
        raise ValueError("%s 不得为空序列" % (name,))
    if len(pair) != 2:
        raise ValueError("%s 必须是长度 2 的 (cu, cv)，得到长度 %d" % (name, len(pair)))
    return (
        _coerce_real(pair[0], name + "[0]"),
        _coerce_real(pair[1], name + "[1]"),
    )


def _is_pair(value):
    """判断是否为 ``speed=(su, sv)`` 形式的二元数值组。"""
    if isinstance(value, (str, bytes, bytearray, dict)) or isinstance(value, _Real):
        return False
    if isinstance(value, (tuple, list)):
        return len(value) == 2
    try:
        pair = tuple(value)
    except TypeError:
        return False
    return len(pair) == 2


def _parse_speed_pair(speed_u, speed_v, speed):
    """把速度写法归一化为 ``(speed_u, speed_v)``。

    支持 ``UvScroll(su, sv)``、``UvScroll((su, sv))`` 与 ``speed=(su, sv)``；
    混用、缺失或类型不符时报 ``TypeError`` / ``ValueError``。
    """
    if speed is not None:
        if speed_u is not None or speed_v is not None:
            raise InvalidTypeError("speed 不能与 speed_u / speed_v 同时指定")
        if not _is_pair(speed):
            raise ValueError(
                "speed 必须是 (speed_u, speed_v) 二元组，得到 %r" % (speed,)
            )
        return tuple(speed)
    if speed_u is None:
        raise InvalidTypeError("缺少速度：请传 UvScroll(su, sv) 或 speed=(su, sv)")
    if speed_v is None:
        if _is_pair(speed_u):
            return tuple(speed_u)
        raise InvalidTypeError("缺少 speed_v：请传 UvScroll(su, sv) 或 speed=(su, sv)")
    return speed_u, speed_v


def _normalize_layers(layers):
    """归一化为 ``[(cu, cv), ...]``；``None`` 表示单层、系数 1.0。"""
    if layers is None:
        return [(1.0, 1.0)]
    if isinstance(layers, bool):
        raise InvalidTypeError("layers 必须是层系数序列，得到 bool")
    if isinstance(layers, _Integral):
        count = int(layers)
        if count < 1:
            raise ValueError("层数必须为正整数，得到 %r" % (count,))
        if count > _MAX_LAYERS:
            raise ValueError("层数不得超过 %d，得到 %r" % (_MAX_LAYERS, count))
        # 整数表示层数：经典视差按 1、1/2、1/4 … 递减生成系数。
        return [(0.5 ** i, 0.5 ** i) for i in range(count)]
    if isinstance(layers, _Real):
        # 单个标量系数：单层。
        c = _coerce_real(layers, "layers")
        return [(c, c)]
    if isinstance(layers, dict):
        # 单个层描述映射 -> 单层；否则视为 {层名: 系数} 的具名多图层。
        recognized = (set(layers) & set(_LAYER_KEYS + _LAYER_U_KEYS + _LAYER_V_KEYS))
        if recognized:
            return [_layer_from_mapping(layers, "layers")]
        if not layers:
            raise ValueError("layers 不得为空映射")
        if len(layers) > _MAX_LAYERS:
            raise ValueError(
                "层数不得超过 %d，得到 %d" % (_MAX_LAYERS, len(layers))
            )
        return [_normalize_layer(value, i) for i, value in enumerate(layers.values())]
    if isinstance(layers, (str, bytes, bytearray)):
        raise InvalidTypeError("layers 必须是层系数序列，得到 %r" % (type(layers).__name__,))
    try:
        items = list(layers)
    except TypeError:
        raise InvalidTypeError(
            "layers 必须是层系数序列，得到 %r" % (type(layers).__name__,)
        ) from None
    if not items:
        raise ValueError("layers 不得为空序列")
    if len(items) > _MAX_LAYERS:
        raise ValueError("层数不得超过 %d，得到 %d" % (_MAX_LAYERS, len(items)))
    return [_normalize_layer(item, i) for i, item in enumerate(items)]


class UvScroll:
    """按速度滚动 UV，可选多层视差，帧索引与 UV 严格同步。"""

    def __init__(self, speed_u=None, speed_v=None, frames=1, layers=None,
                 parallax=None, layer_coeffs=None, coefficients=None,
                 coeffs=None, layer_speeds=None, parallax_coeffs=None,
                 scales=None, layer_scales=None, speed_coeffs=None,
                 layer_factors=None, factors=None, parallax_layers=None,
                 layer_spec=None, layer_config=None, coefficient=None,
                 coeff=None, n_layers=None, num_layers=None,
                 layer_count=None, nlayers=None, speed=None,
                 frame_count=None, num_frames=None, nframes=None,
                 n_frames=None, **options):
        """构造。

        ``speed_u`` / ``speed_v`` 为有限实数（UV/秒，可为负）；也可用
        ``speed=(su, sv)`` 或把 ``speed_u`` 直接传成二元组并省略
        ``speed_v``。``frames`` 为正整数（动画帧数）；``layers`` 为可选的
        层系数：标量表示单层系数，序列表示逐层系数（标量或 ``(cu, cv)``），
        整数表示层数并按 1/2 递减生成系数。各别名（``parallax`` /
        ``layer_coeffs`` / ``coefficients`` / ``coeffs`` / ``layer_speeds``
        / ``parallax_coeffs`` / ``scales`` / ``layer_scales``）都是
        ``layers`` 的等价写法；``n_layers`` / ``num_layers`` 是层数的
        等价写法（可与系数序列同时给出，但必须一致）。
        """
        if options:
            unknown = ", ".join(sorted(options))
            raise InvalidTypeError("不支持的关键字参数：%s" % (unknown,))

        # 兼容 ``UvScroll((su, sv), frames)`` 写法：首个参数为速度二元组时，
        # 第二个位置参数被当作帧数。
        if (_is_pair(speed_u) and speed is None and
                isinstance(speed_v, _Integral) and
                not isinstance(speed_v, bool)):
            if frames != 1:
                raise InvalidTypeError(
                    "帧数不能同时由第 2 位置参数与 frames 给出"
                )
            speed_u, speed_v, frames = speed_u[0], speed_u[1], speed_v
        speed_u, speed_v = _parse_speed_pair(speed_u, speed_v, speed)
        speed_u = _coerce_real(speed_u, "speed_u")
        speed_v = _coerce_real(speed_v, "speed_v")

        frame_aliases = [v for v in (frame_count, num_frames, nframes,
                                     n_frames) if v is not None]
        if len(frame_aliases) > 1:
            raise InvalidTypeError("帧数参数只能指定其一")
        if frame_aliases:
            if frames != 1:
                raise InvalidTypeError("帧数不能同时由 frames 与别名给出")
            frames = frame_aliases[0]
        if isinstance(frames, bool) or not isinstance(frames, _Integral):
            raise InvalidTypeError("frames 必须是正整数，得到 %r" % (type(frames).__name__,))
        frames = int(frames)
        if frames < 1:
            raise ValueError("frames 必须为正整数，得到 %r" % (frames,))

        specs = [v for v in (layers, parallax, layer_coeffs, coefficients,
                             coeffs, layer_speeds, parallax_coeffs, scales,
                             layer_scales, speed_coeffs, layer_factors,
                             factors, parallax_layers, layer_spec,
                             layer_config, coefficient, coeff)
                  if v is not None]
        counts = [v for v in (n_layers, num_layers, layer_count, nlayers)
                  if v is not None]
        for count in counts:
            if isinstance(count, bool) or not isinstance(count, _Integral):
                raise InvalidTypeError(
                    "层数必须是正整数，得到 %r" % (type(count).__name__,)
                )
        if len(specs) > 1:
            raise InvalidTypeError(
                "层系数参数只能指定其一：%s"
                % (" / ".join(_COEFF_ALIASES),)
            )
        if len(counts) > 1:
            raise InvalidTypeError(
                "层数参数只能指定其一：%s" % (" / ".join(_COUNT_ALIASES),)
            )
        if specs:
            coeffs = _normalize_layers(specs[0])
            if counts and int(counts[0]) != len(coeffs):
                raise ValueError(
                    "层数 %r 与系数数量 %d 不一致"
                    % (counts[0], len(coeffs))
                )
        else:
            coeffs = _normalize_layers(counts[0] if counts else None)

        self._frames = frames
        self._base_speed_u = speed_u
        self._base_speed_v = speed_v
        self._coeffs = tuple(coeffs)
        self._su = tuple(speed_u * cu for cu, _ in coeffs)
        self._sv = tuple(speed_v * cv for _, cv in coeffs)
        if not all(map(_isfinite, self._su + self._sv)):
            raise NonFiniteInputError(
                "速度与层系数相乘后溢出为非有限值：speed=(%r, %r), coeffs=%r"
                % (speed_u, speed_v, self._coeffs)
            )
        self._u = [0.0] * len(coeffs)  # 每层 u 相位，恒在 [0, 1)
        self._v = [0.0] * len(coeffs)  # 每层 v 相位，恒在 [0, 1)
        self._max_speed = max((abs(s) for s in self._su + self._sv), default=0.0)
        self._lock = _threading.Lock()

    # ------------------------------------------------------------------ 更新
    def update(self, dt):
        """按 ``dt`` 秒推进所有层。

        非有限或超过可表示范围的 ``dt`` 抛 :class:`NonFiniteInputError`；
        负 ``dt`` 抛 ``ValueError``；非数值抛 ``TypeError``。
        """
        t = type(dt)
        if t is float:
            if not _isfinite(dt):
                raise NonFiniteInputError("dt 必须是有限数值，得到 %r" % (dt,))
            if dt < 0.0:
                raise ValueError("dt 不能为负，得到 %r" % (dt,))
        elif t is int:
            dt = float(dt)
            if dt < 0.0:
                raise ValueError("dt 不能为负，得到 %r" % (dt,))
        else:
            dt = _coerce_real(dt, "dt")
            if dt < 0.0:
                raise ValueError("dt 不能为负，得到 %r" % (dt,))

        if dt != 0.0:
            if not _isfinite(dt * self._max_speed):
                # 相位在 [0, 1) 内，只有速度与 dt 相乘溢出才会产生非有限
                # 相位；在写任何状态之前整体拒绝，保证不留半更新。
                raise NonFiniteInputError(
                    "dt=%r 与速度相乘后溢出为非有限值" % (dt,)
                )

        with self._lock:
            u = self._u
            v = self._v
            su = self._su
            sv = self._sv
            mod = _MOD
            for i in range(len(su)):
                u[i] = (u[i] + su[i] * dt) % mod
                v[i] = (v[i] + sv[i] * dt) % mod

    def reset(self):
        """把各层相位归零（不改变速度、系数与帧数）。"""
        with self._lock:
            for i in range(len(self._u)):
                self._u[i] = 0.0
                self._v[i] = 0.0

    def add_layer(self, coeff):
        """在末尾追加一层，``coeff`` 为标量或 ``(cu, cv)``；返回该层索引。"""
        cu, cv = _normalize_layer(coeff, len(self._coeffs))
        with self._lock:
            if len(self._coeffs) >= _MAX_LAYERS:
                raise ValueError("层数不得超过 %d" % (_MAX_LAYERS,))
            su = self._base_speed_u * cu
            sv = self._base_speed_v * cv
            if not (_isfinite(su) and _isfinite(sv)):
                raise NonFiniteInputError(
                    "速度与层系数相乘后溢出为非有限值：(%r, %r)" % (su, sv)
                )
            self._coeffs = self._coeffs + ((cu, cv),)
            self._su = self._su + (su,)
            self._sv = self._sv + (sv,)
            self._u = self._u + [0.0]
            self._v = self._v + [0.0]
            edge = max(abs(su), abs(sv))
            if edge > self._max_speed:
                self._max_speed = edge
            return len(self._coeffs) - 1

    # ------------------------------------------------------------------ 查询
    def _layer_index(self, layer):
        if type(layer) is int:  # 常见路径：省去 ABC 检查
            index = layer
        elif isinstance(layer, bool) or not isinstance(layer, _Integral):
            raise InvalidTypeError(
                "layer 必须是整数索引，得到 %r" % (type(layer).__name__,)
            )
        else:
            index = int(layer)
        count = len(self._coeffs)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError("层索引越界：%r（共 %d 层）" % (layer, count))
        return index

    def uv(self, layer=0):
        """返回第 ``layer`` 层的当前 UV，形如 ``(u, v)``。"""
        with self._lock:
            index = self._layer_index(layer)
            return (self._u[index], self._v[index])

    def uvs(self):
        """返回所有层的当前 UV：``((u0, v0), (u1, v1), ...)``。"""
        with self._lock:
            return tuple(zip(self._u, self._v))

    def layer_uv(self, layer):
        """``uv(layer)`` 的别名。"""
        return self.uv(layer)

    def layer_uvs(self):
        """``uvs()`` 的别名。"""
        return self.uvs()

    def frame(self, layer=0):
        """返回第 ``layer`` 层由 UV 相位导出的帧索引，``0 <= frame < frames``。"""
        with self._lock:
            index = self._layer_index(layer)
            return self._frame_for(self._u[index])

    def layer_frame(self, layer):
        """``frame(layer)`` 的别名。"""
        return self.frame(layer)

    def frame_index(self, layer=0):
        """``frame(layer)`` 的别名。"""
        return self.frame(layer)

    def layer_frames(self):
        """返回所有层的帧索引。"""
        with self._lock:
            return tuple(self._frame_for(u) for u in self._u)

    def _frame_for(self, phase):
        """帧索引严格由相位导出，保证与 :meth:`uv` 同步。"""
        frame = int(phase * self._frames)
        if frame >= self._frames:  # 浮点边界防护，保持 [0, frames)
            frame = self._frames - 1
        elif frame < 0:
            frame = 0
        return frame

    def layer_count(self):
        """层数。"""
        return len(self._coeffs)

    @property
    def n_layers(self):
        """层数（``layer_count()`` 的属性形式）。"""
        return len(self._coeffs)

    def speed(self, layer=0):
        """返回第 ``layer`` 层的实际速度，形如 ``(du, dv)``。"""
        with self._lock:
            index = self._layer_index(layer)
            return (self._su[index], self._sv[index])

    def layer_coeffs(self):
        """各层速度系数：``((cu, cv), ...)``。"""
        return self._coeffs

    def set_layers(self, layers):
        """整体替换层配置（相位重置为零）；返回新的层数。"""
        coeffs = _normalize_layers(layers)
        su = tuple(self._base_speed_u * cu for cu, _ in coeffs)
        sv = tuple(self._base_speed_v * cv for _, cv in coeffs)
        if not all(map(_isfinite, su + sv)):
            raise NonFiniteInputError("速度与层系数相乘后溢出为非有限值")
        with self._lock:
            self._coeffs = tuple(coeffs)
            self._su = su
            self._sv = sv
            self._u = [0.0] * len(coeffs)
            self._v = [0.0] * len(coeffs)
            self._max_speed = max(
                (abs(s) for s in su + sv), default=0.0
            )
            return len(coeffs)

    def set_speed(self, speed_u, speed_v=None):
        """更新基础速度（保留各层相位与系数）；返回实际 ``(su, sv)``。"""
        speed_u, speed_v = _parse_speed_pair(speed_u, speed_v, None)
        speed_u = _coerce_real(speed_u, "speed_u")
        speed_v = _coerce_real(speed_v, "speed_v")
        su = tuple(speed_u * cu for cu, _ in self._coeffs)
        sv = tuple(speed_v * cv for _, cv in self._coeffs)
        if not all(map(_isfinite, su + sv)):
            raise NonFiniteInputError("速度与层系数相乘后溢出为非有限值")
        with self._lock:
            self._base_speed_u = speed_u
            self._base_speed_v = speed_v
            self._su = su
            self._sv = sv
            self._max_speed = max((abs(s) for s in su + sv), default=0.0)
            return (speed_u, speed_v)

    @property
    def coeffs(self):
        """各层速度系数（``layer_coeffs()`` 的属性形式）。"""
        return self._coeffs

    @property
    def layers(self):
        """各层速度系数（``layer_coeffs()`` 的属性形式）。"""
        return self._coeffs

    @property
    def frames(self):
        """动画帧数。"""
        return self._frames

    @property
    def speed_u(self):
        """基础 u 速度（未乘层系数）。"""
        return self._base_speed_u

    @property
    def speed_v(self):
        """基础 v 速度（未乘层系数）。"""
        return self._base_speed_v


# 语义化别名：多层视差滚动与普通 UV 滚动同构。
ParallaxUvScroll = UvScroll
ParallaxScroll = UvScroll
MultiLayerUvScroll = UvScroll
MultiLayerScroll = UvScroll
