import torch
import matplotlib.pyplot as plt

# 定义x和y
x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = torch.sigmoid(x)

# 创建子图
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(12, 4)

# 1. sigmoid函数土象
ax[0].plot(x.data, y.data, color='purple')
ax[0].set_title('sigmoid(x)')
ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)
ax[0].spines["left"].set_position("zero")
ax[0].spines["bottom"].set_position("zero")
ax[0].axhline(0.5, color="gray", alpha=0.7, linewidth=1)
ax[0].axhline(1, color="gray", alpha=0.7, linewidth=1)

# 反向传播计算梯度
torch.sigmoid(x).sum().backward()
ax[1].plot(x.data, x.grad, "purple")
ax[1].set_title("sigmoid'(x)")
ax[1].spines["top"].set_visible(False)
ax[1].spines["right"].set_visible(False)
ax[1].spines["left"].set_position("zero")
ax[1].spines["bottom"].set_position("zero")

plt.show()