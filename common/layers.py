from common.functions import *
# ReLU层
class Relu:
    def __init__(self):
        # 记录哪些x <= 0
        self.mask = None

    # 向前传播
    def forward(self,x):
        self.mask = (x <= 0)
        y = x.copy()
        y[self.mask] = 0
        return y

    # 反向传播
    def backward(self,dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

# Sigmoid层
class Sigmoid:
    def __init__(self):
        self.y = None

    # 向前传播
    def forward(self,x):
        y = sigmoid(x)
        self.y = y
        return y

    # 反向传播
    def backward(self,dout):
        dx = dout * (1.0 - self.y) * self.y
        return dx

# 仿射层
class Affine:
    def __init__(self, W, b):
        # 保存权重和偏置参数
        self.W = W
        self.b = b
        self.X = None
        self.X_original_shape = None
        self.dW = None
        self.db = None

    # 向前传播
    def forward(self, X):
        self.X_original_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)
        y = np.dot(self.X, self.W) + self.b
        return y

    # 反向传播
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        # 计算参数的梯度
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx.reshape(*self.X_original_shape)