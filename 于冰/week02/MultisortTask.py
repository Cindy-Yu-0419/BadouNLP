"""

基于pytorch框架编写模型训练
实现一个自行构造的分类(机器学习)任务
规律：随机生成五维随机向量，维度大的维度作为类别

"""
import torch as torch
import torch.nn as  nn
import numpy as np
import matplotlib.pyplot as plt

#定义模型
class TorchModel (nn.Module):
    def __init__(self,input_size,category_num):
        super(TorchModel, self).__init__()
        # input_size 输入样本大小，category_num表示输出的类别数量
        self.linear = nn.Linear(input_size,category_num) #线性层
        self.activation = nn.functional.sigmoid #sigmoid归一化函数
        self.loss = nn.functional.cross_entropy #交叉熵 作为损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x,y = None): #forward 正向传播。backward 反向传播
        x = self.linear(x)
        y_pre = self.activation(x) #y_pre 预测值
        if y is None:
            return y_pre.argmax(dim = 1) #返回预测结果
        else:
            return self.loss(y_pre,y) #预测值和真实值 计算其损失

# 随机生成样本--随机生成五维随机向量，维度大的维度作为类别
def build_target():
    data = torch.randn(5)  # 五维随机数据
    category = np.argmax(data) #确定类别 最大维度的索引
    return data, category
# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range (total_sample_num):
        x,y = build_target()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))

#测试模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pre = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pre, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size,2)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size,2)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果

if __name__ == "__main__":
    main()