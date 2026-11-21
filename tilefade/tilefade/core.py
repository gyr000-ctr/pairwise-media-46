"""tilefade 核心 API（初始未实现）。"""


class TileFade:
    def __init__(self, duration):
        """构造"""
        raise NotImplementedError

    def update(self, dt):
        """更新"""
        raise NotImplementedError

    def alpha(self):
        """当前透明度"""
        raise NotImplementedError
