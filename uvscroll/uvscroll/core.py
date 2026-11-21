"""uvscroll 核心 API（初始未实现）。"""


class UvScroll:
    def __init__(self, speed_u, speed_v, frames=1):
        """构造"""
        raise NotImplementedError

    def update(self, dt):
        """更新"""
        raise NotImplementedError

    def uv(self):
        """当前 UV"""
        raise NotImplementedError

    def frame(self):
        """当前帧"""
        raise NotImplementedError
