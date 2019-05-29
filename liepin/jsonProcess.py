# encoding=utf-8
import jieba
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
import tensorflow as tf

stop_words = ['，','呀','是','只','都','怎么样','可以','自己','咱们','就','吗','的','是不是','人','怎么','说','跟','不想','好','这方面','话','一定','肯定',
              '大','想','要','这','你','有没有','我','和','几个','了','找','下','这样','很','那','只有','能','什么样','你们','会','这个','有','一下','什么','没有','一般',
              '不会','呢','啥','还是','吧','的话','啊','请问','问','请','想问',]

def delete_stop_words(sentence):
    tem = list(jieba.cut(sentence))
    for i in tem:
        if i in stop_words:
            tem.remove(i)
    return tem


with open('liepin_data.json', encoding='utf-8') as f:
    data = f.read()
    data_dict = eval(data)

f = open('tem.csv', 'w', encoding='utf-8')
f.write("words,label\n")
for count, intent in enumerate(data_dict):
    for sentence in data_dict[intent]:
        sentence = delete_stop_words(sentence)
        line = '/'.join(sentence)+','+str(count)+'\n'
        f.write(line)

f.close()
f = pd.read_csv('tem.csv')
f = shuffle(f)
x_data, y_data = f.words, f.label

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.1)
x_train, x_valid,  y_train, y_valid = train_test_split(
    x_train, y_train, test_size=0.11111)

train = pd.DataFrame({'label': y_train, 'x_train': x_train})
train.to_csv("data/train.csv", index=False, sep='\t')
valid = pd.DataFrame({'label': y_valid, 'x_train': x_valid})
valid.to_csv("data/dev.csv", index=False, sep='\t')
test = pd.DataFrame({'label': y_test, 'x_test': x_test})
test.to_csv("data/test.csv", index=False, sep='\t')
