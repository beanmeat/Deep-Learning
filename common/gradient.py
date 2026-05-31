import numpy as np


# 数值微分，传入函数f和自变量x
def numerical_diff0(f, x):
    h = 1e-4  # 微小值 0.0001
    return (f(x + h) - f(x)) / h


# 中心差分实现数值微分
def numerial_diff(f, x):
    h = 1e-4  # 微小值 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


# 利用数值微分计算梯度
# f：多元函数，x：向量 [x1,x2,...,xn]
def _numerical_gradient(f, x):
    # 定义微小量
    h = 1e-4  # 0.0001
    # 定义梯度向量，初始值为0
    grad = np.zeros_like(x)

    # 遍历x中的每个变元 xi
    for idx in range(x.size):
        tmp_val = x[idx]  # 临时保存xi，后面需要更改
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 计算对xi的偏微分

        x[idx] = tmp_val  # 还原值

    return grad

# 数值微分计算梯度，扩展到二维矩阵形式
def numerical_gradient(f, X):
    # 分一维和二维
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # 遍历X中的每一行，分别求梯度向量
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient(f, x)
        return grad