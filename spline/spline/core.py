"""spline 核心 API（初始未实现）。"""


class Spline:
    def __init__(self, points):
        """按控制点构造"""
        raise NotImplementedError

    def sample(self, t):
        """按参数采样"""
        raise NotImplementedError

    def resample(self, spacing):
        """等距重采样"""
        raise NotImplementedError
