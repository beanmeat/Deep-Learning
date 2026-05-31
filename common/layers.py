# ReLU层
class Relu:
    def __init__(self):
        # 记录哪些x <= 0
        self.mask = None

    # 向前传播
    def forward(self,x):