import numpy as np
# 1. 阶跃函数
# x传入标量
def step_function0(x):
    if x >= 0:
        return 1
    else:
        return 0

# x 传入向量或矩阵
def step_function1(x):
    return np.array(x >= 0, dtype=int)

# 2. Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3. ReLU函数
def relu(x):
    return np.maximum(0, x)

# 4. Softmax函数
def softmax0(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

# 5. 恒等函数
def identity(x):
    return x

# 二、损失函数
# 1. MSE
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 2.交叉熵误差
def cross_entropy_error(y, t):
    # 对于一维情况，直接转换为二维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # t是独热编码表示，转换为正确类别的索引
    if t.size == y.size:
        t = t.argmax(axis = 1)
    n = y.shape[0]
    return -np.sum(np.log(y[np.arange(n), t]))

# 测试
x = np.array([0,1,2,3,4,5,-1,-2,-3,-4,-5])