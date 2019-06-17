import numpy as np
import pandas as pd
import random
from keras.utils import to_categorical
from MAML import MAML
def resample(data_dict,sample_size,seed,test_size):
    """
    数据重采样
    :param data_dict: 输入数据dict类型：{label_1:[text_11,text_12,...,text_1j,...],label_2:[...],...,label_i:[...],...}
    :param sample_size: 每类样本容量：超过的样本类进行下采样，不足的样本类进行上采样
    :param seed: 随机种子
    :param test_rate: 测试集数量
    :return: train：训练集；test：测试集
    """
    random.seed(seed)
    np.random.seed(seed)
    train = []
    test = []
    for label in data_dict:
        #先抽取测试集
        num_data = len(data_dict[label])
        test_num_data = test_size
        for i in range(test_num_data):
            test_data = np.random.choice(data_dict[label],1)
            test_index = np.where(data_dict[label]==test_data)
            if test_num_data > 1:
                data_dict[label] = np.delete(data_dict[label],test_index)
            test.append([test_data[0],label])
        #再抽取训练集
        if len(data_dict[label]) >= sample_size:
            #下采样(不能重复，要用random.sample)
            train_data = random.sample(list(data_dict[label]),sample_size)
            train.extend([[train_data_i,label] for train_data_i in train_data])
        else:
            #上采样（有重复，要用np.random.choice+原数据）
            train_data = list(data_dict[label])
            train_data_extend = np.random.choice(train_data,sample_size-len(train_data)).tolist()
            train_data.extend(train_data_extend)
            train.extend([[train_data_i,label] for train_data_i in train_data])
    return train,test


if __name__ == '__main__':
    # 1. 数据加载
    file_path = './data/liepin.xlsx'
    file_df = pd.read_excel(file_path, usecols=[0,1], names=['intent', 'text'])
    data = file_df.text.values
    label = file_df.intent.values

    data_dict = {}
    for data_i, label_i in zip(data,label):
        data_dict.setdefault(label_i,[]).append(data_i)
    train, test = resample(data_dict=data_dict, sample_size=5, seed=0, test_size=1)

    # 2. 获取bert向量
    from bert_serving.client import BertClient
    bc = BertClient()

    for line in train:
        line[0] = bc.encode([line[0]])[0]
    for line in test:
        line[0] = bc.encode([line[0]])[0]

    # 3. 将label类别化
    label_set = []
    for label in test:
        if label[1] not in label_set:
            label_set.append(label[1])

    label_convert = dict([[item,label_set.index(item)] for item in label_set])

    train_label = [label_convert[item[1]] for item in train]
    test_label = [label_convert[item[1]] for item in test]
    train_label = to_categorical(train_label)
    test_label = to_categorical(test_label)

    maml = MAML(num_tasks=76, num_samples=5, num_epoch=12, alhpa=0.0001, beta=0.0001)
    maml.train(train, test, train_label, test_label)
