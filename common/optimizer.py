import numpy as np

# 随机梯度下降
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# 动量法
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

# AdaGrad
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# RMSProp
class RMSProp:
    def __init__(self, lr=0.01, alpha=0.9):
        self.lr = lr
        self.alpha = alpha
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] = self.alpha * self.h[key] + (1 - self.alpha) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

# Adam
class Adam:
    def __init__(self, lr=0.01, alpha1=0.9, alpha2=0.999):
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.t = 0
        self.h = None
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v, self.h = {}, {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        self.t += 1
        # 用系数对学习率进行调整
        lr_t = self.lr * np.sqrt(1.0 - self.alpha2 ** self.t) / (1.0 - self.alpha1 ** self.t)
        # 更新参数
        for key in params.keys():
            self.v[key] = self.alpha1 * self.v[key] + (1 - self.alpha1) * grads[key]
            self.h[key] = self.alpha2 * self.h[key] + (1 - self.alpha2) * grads[key] * grads[key]
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-7)
