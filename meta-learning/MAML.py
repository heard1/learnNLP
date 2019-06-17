import numpy as np


class MAML(object):
    def __init__(self, num_tasks, num_samples, num_epoch, alhpa, beta):
        # 每一个batch需要的task数
        self.num_tasks = num_tasks
        # 每一个class需要的sample数，N-shot
        self.num_samples = num_samples
        # 训练epoch数
        self.epochs = num_epoch
        # 第一次梯度下降学习率
        self.alpha = alhpa
        # 第二次梯度下降学习率
        self.beta = beta
        # 随机初始化模型
        self.theta = np.random.normal(size=768).reshape(768, 1)

    # 激活函数使用sigmoid
    def sigmoid(self, a):
        return 1.0 / (1 + np.exp(-a))

    def train(self, train, test, train_label, test_label):
        train_x, train_y = np.array([item[0] for item in train]), train_label
        test_x, test_y = np.array([item[0] for item in test]), test_label
        for e in range(self.epochs):
            self.theta_ = []
            # 每一个task
            for i in range(self.num_tasks):
                # 任意获取train和label
                XTrain, YTrain = [], []
                for i in range(5):
                    index = np.random.randint(380)
                    XTrain.append(train_x[index])
                    YTrain.append(train_y[index])
                XTrain = np.array(XTrain)
                YTrain = np.array(YTrain)
                tem = []
                for i in range(5):
                    tem.append(int(np.argwhere(YTrain[i] == 1))/76)
                YTrain = np.array([tem]).T
                a = np.matmul(XTrain, self.theta)
                YHat = self.sigmoid(a)
                # 交叉熵损失函数
                loss = \
                ((np.matmul(-YTrain.T, np.log(YHat)) - np.matmul((1 - YTrain.T), np.log(1 - YHat))) / self.num_samples)[
                    0][0]
                # 计算梯度
                gradient = np.matmul(XTrain.T, (YHat - YTrain)) / self.num_samples
                # 更新梯度
                self.theta_.append(self.theta - self.alpha * gradient)
            # 更新meta梯度
            meta_gradient = np.zeros(self.theta.shape)
            for i in range(self.num_tasks):
                # 获取测试集
                XTest = test_x
                tem = []
                for i in range(len(test_y)):
                    tem.append(int(np.argwhere(test_y[i] == 1))/76)
                YTest = np.array([tem]).T
                # 计算梯度
                a = np.matmul(XTest, self.theta_[i])
                YPred = self.sigmoid(a)
                meta_gradient += np.matmul(XTest.T, (YPred - YTest)) / self.num_samples

            # 更新模型
            self.theta = self.theta - self.beta * meta_gradient / self.num_tasks

            if True:
                print("Epoch {}: Loss {}".format(e, loss))
                print('Updated Model Parameter Theta')
                print('Sampling Next Batch of Tasks')
                print('---------------------------------')
