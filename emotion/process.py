import xml.dom.minidom as xmldom
import os
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import requests
import json
import csv
from keras.utils.np_utils import to_categorical
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D, GlobalAveragePooling1D

def getXML():
    list_datafile = os.listdir('./data')
    neg, pos = 0, 0
    for single_XML in list_datafile:
        dom = xmldom.parse("data/"+single_XML)
        sub = dom.getElementsByTagName('sentence')
        for single in sub:
            with open('output/output.txt', 'a', encoding='utf-8') as f:
                if single.getAttribute('polarity') == 'NEG':
                    line = '2,' + single.firstChild.data + '\n'
                    f.write(line)
                    neg += 1
                elif single.getAttribute('polarity') == 'POS':
                    line = '1,' + single.firstChild.data + '\n'
                    f.write(line)
                    pos += 1
    print(pos, neg)


def get_git():
    f = open('data/train.txt', encoding='utf-8')
    target = open('output/output.txt', 'a', encoding='utf-8')
    list_f = f.readlines()
    for sentence in list_f:
        if sentence[0] == '1':
            line = '1,' + sentence[2:]
            target.write(line)
        elif sentence[0] == '2':
            line = '2,' + sentence[2:]
            target.write(line)
    f.close()
    target.close()


def clean():
    regex1 = re.compile('#.*?#')
    regex2 = re.compile(u'[^\u4E00-\u9FA5]')
    new = open('output/output2.txt', 'a', encoding='utf-8')
    with open('output/output.txt', encoding='utf-8') as f:
        for sentence in f.readlines():
            new_line = sentence[:2] + regex2.sub('', regex1.sub('', sentence[2:])) + '\n'
            new.write(new_line)
    new.close()


def divide(filename):
    regex = re.compile(u'[^\u4E00-\u9FA5]')
    new = open('chnsenticorp/output3.txt', 'a', encoding='utf-8')
    stop_words = ['是', '在', '呀', '又', '太', '是', '只', '都', '怎么样', '可以', '自己', '咱们', '就', '吗', '的', '是不是', '人', '怎么', '说', '跟', '不想',
                  '好', '这方面', '话', '一定', '肯定',
                  '大', '想', '要', '这', '你', '有没有', '我', '和', '几个', '了', '找', '下', '这样', '很', '那', '只有', '能', '什么样', '你们',
                  '会', '这个', '有', '一下', '什么', '没有', '一般',
                  '不会', '呢', '啥', '还是', '吧', '的话', '啊', '请问', '问', '请', '想问', '我们', '你', '他']
    with open('chnsenticorp/'+filename, encoding='utf-8') as f:
        f.readline()
        for line in f.readlines():
            tem = list(jieba.cut(regex.sub('', line[2:])))
            for i in tem:
                if i in stop_words:
                    tem.remove(i)
            new_line = line[:2] + '/'.join(tem)+'\n'
            new.write(new_line)
    new.close()


def generate():
    f = pd.read_csv('output.csv')
    f = shuffle(f)
    x_data, y_data = f.words, f.label
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_data, y_data, test_size=0.1)
    train = pd.DataFrame({'label': y_train, 'x_train': x_train})
    train.to_csv("train.csv", index=False, sep='\t')
    valid = pd.DataFrame({'label': y_valid, 'x_train': x_valid})
    valid.to_csv("dev.csv", index=False, sep='\t')


def get_bert_encode(texts):
    r = requests.post("http://192.168.110.8:5003/encode", json={
                "id": 123,
                "texts": texts,
                "is_tokenized": False
            })
    result = json.loads(r.text)  # str --> json
    return np.asarray(result['result'])
# bert-serving-start -model_dir uncased_L-12_H-768_A-12 -num_worker=4
#from bert_serving.client import BertClient
#bc = BertClient()
def sentence2vec(sentence):
    embedding_matrix = np.zeros((100, 768))
    tem = sentence.split('/')
    for count, i in enumerate(tem):
        if i != '':
            try:
                embedding_matrix[count] = bc.encode([i])[0]
            except:
                print(i)
    return embedding_matrix


def getData():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    with open('train.csv', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        i = 0
        for line in reader:
            if i == 0:
                i += 1
                continue
            y_train.append(line[0])
            x_train.append(sentence2vec(line[1]))
    with open('dev.csv', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        i = 0
        for line in reader:
            if i == 0:
                i += 1
                continue
            y_valid.append(line[0])
            x_valid.append(sentence2vec(line[1]))
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_valid.npy', x_valid)
    np.save('y_valid.npy', y_valid)
    return x_train, y_train, x_valid, y_valid

# 89%
def CNN(x_train, y_train, x_valid, y_valid):
    input = Input(shape=(100, 768, ), dtype='float32')
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(input)
    cnn1 = MaxPooling1D()(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(input)
    cnn2 = MaxPooling1D()(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(input)
    cnn3 = MaxPooling1D()(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    output = Dense(2, activation='softmax')(drop)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=4, validation_data=(x_valid, y_valid))
    print(history.history)
    return model

from sklearn.svm import SVC #83%
from sklearn.ensemble import RandomForestClassifier #81.4%
from sklearn.ensemble import GradientBoostingClassifier #87.5%
from sklearn.linear_model import SGDClassifier  #90%

def SGD(x_train, y_train, x_valid, y_valid):
    X_train = [np.sum(x, axis=0) for x in x_train]
    X_train = np.array(X_train)
    X_valid = [np.sum(x, axis=0) for x in x_valid]
    X_valid = np.array(X_valid)
    Y_train = np.argwhere(y_train == 1)
    Y_train = [x[1] for x in Y_train]
    Y_valid = np.argwhere(y_valid == 1)
    Y_valid = [x[1] for x in Y_valid]
    sgd = SGDClassifier()
    sgd.fit(X_train, Y_train)
    sgd.score(X_valid, Y_valid)