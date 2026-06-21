import torch # 张量
from torch import nn, optim # 引入神经网络模块和优化器
from torch.utils.data import DataLoader, TensorDataset  # 数据集和数据加载器
import matplotlib.pyplot as plt # 画图

# 1. 构建数据
x = torch.randn(100,1)
w = torch.tensor([2.5])
b = torch.tensor([5.2])
# 引入噪声
noise = torch.randn(100,1) * 0.5
y = w * x + b + noise

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2. 构建模型
model = nn.Linear(1, 1)

# 3. 定义损失函数和优化器
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义列表保存训练误差
loss_list = []
# 4. 训练模型
for epoch in range(100):
    # 定义训练误差和训练迭代次数
    total_loss = 0
    iter_num = 0
    for x_train, y_train in dataloader:
        # 前向传播
        y_pred = model(x_train)
        # 计算损失
        loss_value = loss(y_pred, y_train)
        total_loss += loss_value.item()
        iter_num += 1
        # 清除梯度
        optimizer.zero_grad()
        # 反向传播
        loss_value.backward()
        # 更新参数
        optimizer.step()

    # 计算本轮的（epoch）的平均训练误差
    loss_list.append(total_loss / iter_num)

print(model.weight, model.bias) # 打印权重和偏置
plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.scatter(x, y)
y_pred = model.weight.item() * x + model.bias.item()
plt.plot(x, y_pred, color="red")
plt.show()