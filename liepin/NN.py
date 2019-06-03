from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Conv1D, Flatten, Dropout, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import csv
from bert_serving.client import BertClient
#bc = BertClient()

def sentence2vec(sentence):
    embedding_matrix = np.zeros((10, 768))
    tem = sentence.split('/')
    for count, i in enumerate(tem):
        if i != '':
            try:
                embedding_matrix[count] = bc.encode([i])[0]
            except:
                print(i)
    return embedding_matrix

def getData():
    x_train=[]
    y_train=[]
    x_valid=[]
    y_valid=[]
    with open('data/train.csv', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        i = 0
        for line in reader:
            if i==0:
                i+=1
                continue
            y_train.append(line[0])
            x_train.append(sentence2vec(line[1]))
    with open('data/dev.csv', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        i = 0
        for line in reader:
            if i==0:
                i+=1
                continue
            y_valid.append(line[0])
            x_valid.append(sentence2vec(line[1]))
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)
    return np.array(x_train), y_train, np.array(x_valid), y_valid


def train():
    input = Input(shape=(10, 768, ), dtype='float32')
    # 词窗大小分别为3,4,5
    cnn1 = Conv1D(256, 3, padding='same', strides=1, activation='relu')(input)
    cnn1 = MaxPooling1D(pool_size=10)(cnn1)
    cnn2 = Conv1D(256, 4, padding='same', strides=1, activation='relu')(input)
    cnn2 = MaxPooling1D(pool_size=9)(cnn2)
    cnn3 = Conv1D(256, 5, padding='same', strides=1, activation='relu')(input)
    cnn3 = MaxPooling1D(pool_size=8)(cnn3)
    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    drop = Dropout(0.2)(flat)
    output = Dense(76, activation='softmax')(drop)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    # model.fit(x_train, y_train,
    #           batch_size=32,
    #           epochs=8,
    #           validation_data=(x_valid, y_valid))

import jieba
stop_words = ['，','呀','是','只','都','怎么样','可以','自己','咱们','就','吗','的','是不是','人','怎么','说','跟','不想','好','这方面','话','一定','肯定',
              '大','想','要','这','你','有没有','我','和','几个','了','找','下','这样','很','那','只有','能','什么样','你们','会','这个','有','一下','什么','没有','一般',
              '不会','呢','啥','还是','吧','的话','啊','请问','问','请','想问',]

def pre(sentence):
    tem = list(jieba.cut(sentence))
    for i in tem:
        if i in stop_words:
            tem.remove(i)
    matrix = sentence2vec('/'.join(tem))
    m=np.array([matrix])
    return m

from sklearn import svm
def SVM_liepin():
    X_train = [np.sum(x, axis=0) for x in x_train]
    X_train = np.array(X_train)
    X_valid = [np.sum(x, axis=0) for x in x_valid]
    X_valid = np.array(X_valid)
    Y_train = np.argwhere(y_train == 1)
    Y_train = [x[1] for x in Y_train]
    Y_valid = np.argwhere(y_valid == 1)
    Y_valid = [x[1] for x in Y_valid]
    clf = svm.SVC()
    clf.fit(X_train, Y_train)
    clf.predict(X_valid)
    clf.score(X_valid, Y_valid)