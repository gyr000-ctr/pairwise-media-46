"""camerashake 核心 API（初始未实现）。"""


class CameraShake:
    def __init__(self, strength, duration, seed=0):
        """构造"""
        raise NotImplementedError

    def update(self, dt):
        """推进并返回偏移"""
        raise NotImplementedError

    def remaining(self):
        """剩余强度"""
        raise NotImplementedError
