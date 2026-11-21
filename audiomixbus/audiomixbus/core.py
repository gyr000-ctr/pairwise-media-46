"""audiomixbus 核心 API（初始未实现）。"""


class MixBus:
    def __init__(self, send=1.0):
        """构造"""
        raise NotImplementedError

    def push(self, samples):
        """推入"""
        raise NotImplementedError

    def mix(self):
        """取出总线输出"""
        raise NotImplementedError
