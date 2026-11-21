"""colorblind 核心 API（初始未实现）。"""


class ColorBlind:
    def __init__(self, kind):
        """按类型构造"""
        raise NotImplementedError

    def apply(self, color):
        """单色转换"""
        raise NotImplementedError

    def apply_image(self, image):
        """整图转换"""
        raise NotImplementedError
