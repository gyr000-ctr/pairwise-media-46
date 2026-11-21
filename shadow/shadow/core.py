"""shadow 核心 API（初始未实现）。"""


class Shadow:
    def __init__(self, light):
        """按光源构造"""
        raise NotImplementedError

    def project(self, points):
        """生成阴影多边形"""
        raise NotImplementedError

    def area(self):
        """阴影面积"""
        raise NotImplementedError
