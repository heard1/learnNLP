"""
***用ELMo模型提取文本特征***
用了elmo把推文变成向量
    先洗数据
    然后用elmo把文中每个词变成1024维向量
    取平均值得到句子的向量
用逻辑回归做二分类
    输入是1024*1矩阵
"""
import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import tensorflow_hub as hub
import tensorflow as tf

pd.set_option('display.max_colwidth', 200)
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
train = pd.read_csv("C:/Users/Administrator/Desktop/train.csv")
test = pd.read_csv("C:/Users/Administrator/Desktop/test.csv")

train['clean_tweet'] = train['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
test['clean_tweet'] = test['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))


# remove punctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

# convert text to lowercase
train['clean_tweet'] = train['clean_tweet'].str.lower()
test['clean_tweet'] = test['clean_tweet'].str.lower()

# remove numbers
train['clean_tweet'] = train['clean_tweet'].str.replace("[0-9]", " ")
test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

# remove whitespaces
train['clean_tweet'] = train['clean_tweet'].apply(lambda x: ' '.join(x.split()))
test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))

# import spaCy's language model
nlp = spacy.load('en', disable=['parser', 'ner'])


# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output


train['clean_tweet'] = lemmatization(train['clean_tweet'])
test['clean_tweet'] = lemmatization(test['clean_tweet'])


def elmo_vectors(x):
    embeddings = elmo(x.tolist(), signature="default", as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings, 1))


list_train = [train[i:i+100] for i in range(0, train.shape[0], 100)]
list_test = [test[i:i+100] for i in range(0, test.shape[0], 100)]
# Extract ELMo embeddings
elmo_train = [elmo_vectors(x['clean_tweet']) for x in list_train]
elmo_test = [elmo_vectors(x['clean_tweet']) for x in list_test]
# 将它们整合成一个数组：
elmo_train_new = np.concatenate(elmo_train, axis=0)
elmo_test_new = np.concatenate(elmo_test, axis=0)

from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, train['label'], random_state=42, test_size=0.2)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

lreg = LogisticRegression()
lreg.fit(xtrain, ytrain)
# 验证集
preds_valid = lreg.predict(xvalid)
print(f1_score(yvalid, preds_valid))
# 测试集
preds_test = lreg.predict(elmo_test_new)
sub = pd.DataFrame({'id':test['id'], 'label':preds_test})
sub.to_csv("sub_lreg.csv", index=False)
