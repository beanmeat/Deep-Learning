import torch
import torch.nn as nn

# 实现自定义神经网络类
class Model(nn.Module):
    # 初始化
    def __init__(self):
        super(Model, self).__init__()
        # 定义全连接层，并做参数初始化
        self.linear1 = nn.Linear(3, 4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(4, 4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.out = nn.Linear(4, 2)

    def forward(self, x):
        # 隐藏层1
        x = torch.tanh(self.linear1(x))
        # 隐藏层2
        x = self.linear2(x)
        x = torch.relu(x)
        # 输出层
        x = self.out(x)
        x = torch.softmax(x, dim=1)
        return x

# 定义输入数据
x = torch.randn(10, 3)

# 创建模型（三层神经网络）
model = Model()

# 前向传播
output = model.forward(x)

print(output)
