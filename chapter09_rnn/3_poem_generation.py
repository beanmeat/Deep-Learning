import re
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


# 数据预处理
def process_poems(file_path):
    poems = []  # 保存处理后的诗
    char_set = set()  # 保存所有不重复的字
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 逐行处理
            line = re.sub(r"[，。、？！：]", "", line).strip()  # 去掉标点符号与两侧空白
            # 按字保存诗
            poems.append(list(line))
            # 按字分割并去重
            char_set.update(list(line))
    # 构建词表
    vocab = list(char_set) + ["<UNK>"]
    # 创建词到索引的映射
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    # 将诗转换为索引序列
    id_sequences = []
    for poem in poems:
        seq = [word2idx.get(word) for word in poem]
        id_sequences.append(seq)
    return id_sequences, word2idx, vocab


id_sequences, word2idx, vocab = process_poems("../data/poems.txt")


# print(id_sequences)

# 2. 定义训练数据集Dataset
class PoetryDataset(Dataset):
    def __init__(self, id_sequences, seq_len):
        self.data = []
        self.seq_len = seq_len
        # 遍历诗的id列表，截取长度为L的序列x和“后续”序列y
        for seq in id_sequences:
            # 遍历当前诗（seq）的每一个字
            for i in range(0, len(seq) - self.seq_len - 1):
                self.data.append((seq[i:i + self.seq_len], seq[i + 1:i + self.seq_len + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从data中获取序列数据，转换成张量
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y


dataset = PoetryDataset(id_sequences, seq_len=24)


# 3. 搭建模型
class PoetryRNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=1):
        super().__init__()
        # 定义模型中的层，嵌入层，RNN，全链接层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hx=None):
        embedded = self.embedding(input)
        output, hn = self.rnn(embedded, hx)
        output = self.linear(output)
        return output, hn


# 定义模型
model = PoetryRNNLM(len(vocab), embedding_dim=256, hidden_size=512, num_layers=2)


# 4. 模型训练
def train(model, dataset, lr, epoch_num, batch_size, device):
    # 4.1 初始化相关操作
    model.to(device)
    model.train()
    loss = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr) # 优化器

    # 4.2 迭代训练
    for epoch in range(epoch_num):
        loss_total = 0.0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # 小批量梯度下降更新参数
        for batch_count,(x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output,_ = model(x) # 前向传播
            loss_value = loss(output.transpose(1,2),y) # 计算损失
            loss_value.backward() # 反向传播
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 梯度清零
            loss_total += loss_value.item() * x.shape[0]
            print(f"\repoch:{epoch:0>3}[{'=' * (int((batch_count+1) / len(dataloader) * 50)):<50}]", end="")
        # 计算训练平均损失
        print(f"loss:{loss_total/len(dataset):.6f}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, dataset, lr=0.01, epoch_num=1, batch_size=1, device=device)

# 5. 生成文本
def generate_poem(model, word2idx, vocab, start_token, line_num=4, line_length=7):
    model.eval() # 设置为预测模式
    poem = [] # 记录生成结果
    current_line_length = line_length # 当前句的剩余长度
    start_token = word2idx.get(start_token, word2idx["<UNK>"]) # 起始token
    # 如果起始token在词典中，添加到结果中
    if start_token != word2idx["<UNK>"]:
        poem.append(vocab[start_token])
        current_line_length -= 1
        # 定义神经网络输入数据
        input = torch.LongTensor([[start_token]]).to(device) # 输入
        hidden = None  # 初始化隐状态
        with torch.no_grad(): # 关闭梯度计算
            for _ in range(line_num):   # 生成的行数
                for interpunction in ["，", "。\n"]:   # 每行两句
                    while current_line_length > 0:   # 每句诗line_length个字
                        output, hidden = model(input, hidden)
                        prob = torch.softmax(output[0, 0], dim=-1)   # 计算概率
                        next_token = torch.multinomial(prob, 1)   # 从概率分布中随机采样
                        poem.append(vocab[next_token.item()])   # 将采样结果添加到结果中
                        input = next_token.unsqueeze(0)
                        current_line_length -= 1
                    current_line_length = line_length
                    poem.append(interpunction)   # 每句结尾添加标点符号
            return "".join(poem)   # 将列表转换为字符串




print(generate_poem(model, word2idx, vocab, start_token="一", line_num=4, line_length=7))