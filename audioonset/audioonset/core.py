"""audioonset 核心 API（初始未实现）。"""


class OnsetDetector:
    def __init__(self, rate, threshold, min_gap):
        """构造"""
        raise NotImplementedError

    def process(self, samples):
        """返回起音位置列表"""
        raise NotImplementedError
