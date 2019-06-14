import jieba
import re
import keras
from keras.layers import *
from keras import Input
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, Flatten, Embedding, Dropout, concatenate, \
    Bidirectional, LSTM, GRU, Reshape
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.callbacks import ModelCheckpoint

regex = re.compile(u'[^\u4E00-\u9FA5|0-9a-zA-Z]')


def remove_punctuation(s):
    """
    文本标准化：仅保留中文字符、英文字符和数字字符
    :param s: 输入文本（中文文本要求进行分词处理）
    :return: s_：标准化的文本
    """
    s_ = regex.sub('', s)
    return s_


def text_tokenizer(texts):
    """
    文本分词+标准化
    :param texts: 输入文本list类型：[text_1,text_2,...,text_i,...]
    :return: 标准化后的分词文本
    """
    return [jieba.lcut(remove_punctuation(text)) for text in texts]


f = open("atec_nlp_sim_train2.csv", 'r')
sentence1 = []
sentence2 = []
label = []
for single in f.readlines():
    tem = single[:-1].split('\t')
    sentence1.append(tem[1])
    sentence2.append(tem[2])
    label.append(tem[3])

train1, train2 = sentence1[:-10000], sentence2[:-10000]
test1, test2 = sentence1[len(sentence1) - 10000:], sentence2[len(sentence1) - 10000:]
train_label = label[:-10000]
test_label = label[len(sentence1) - 10000:]

train_data1 = []
train_data2 = []
test_data1 = []
test_data2 = []
for i in train1:
    train_data1.append(text_tokenizer([i])[0])
for i in train2:
    train_data2.append(text_tokenizer([i])[0])
for i in test1:
    test_data1.append(text_tokenizer([i])[0])
for i in test2:
    test_data2.append(text_tokenizer([i])[0])
label_set = []
for label in train_label:
    if label not in label_set:
        label_set.append(label)
label_convert = dict([[item, label_set.index(item)] for item in label_set])
print(label_convert)
label_reconvert = dict([(label_convert[key], key) for key in label_convert])
print(label_reconvert)
train_label = [label_convert[item] for item in train_label]
test_label = [label_convert[item] for item in test_label]
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

num_words = 10000
maxlen = 25
embedding_dim = 128
output_categories = len(label_convert)
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_data1 + train_data2)

train_data1 = tokenizer.texts_to_sequences(train_data1)
test_data1 = tokenizer.texts_to_sequences(test_data1)
train_data1 = pad_sequences(train_data1, maxlen=maxlen)
test_data1 = pad_sequences(test_data1, maxlen=maxlen)
train_data2 = tokenizer.texts_to_sequences(train_data2)
test_data2 = tokenizer.texts_to_sequences(test_data2)
train_data2 = pad_sequences(train_data2, maxlen=maxlen)
test_data2 = pad_sequences(test_data2, maxlen=maxlen)

maxlen = 25
num_words = 10000
embedding_dim = 128

inputs1 = Input(shape=(maxlen,))
inputs2 = Input(shape=(maxlen,))
embedding1 = Embedding(input_dim=num_words, input_length=maxlen, output_dim=embedding_dim, trainable=True)(inputs1)
embedding2 = Embedding(input_dim=num_words, input_length=maxlen, output_dim=embedding_dim, trainable=True)(inputs2)
BiLSTM1 = Bidirectional(LSTM(units=300, return_sequences=True))(embedding1)
BiLSTM2 = Bidirectional(LSTM(units=300, return_sequences=True))(embedding2)

e = Dot(axes=2)([BiLSTM1, BiLSTM2])
e1 = Softmax(axis=2)(e)
e2 = Softmax(axis=1)(e)
e1 = Lambda(K.expand_dims, arguments={'axis': 3})(e1)
e2 = Lambda(K.expand_dims, arguments={'axis': 3})(e2)

_x1 = Lambda(K.expand_dims, arguments={'axis': 1})(BiLSTM2)
_x1 = Multiply()([e1, _x1])
_x1 = Lambda(K.sum, arguments={'axis': 2})(_x1)
_x2 = Lambda(K.expand_dims, arguments={'axis': 2})(BiLSTM1)
_x2 = Multiply()([e2, _x2])
_x2 = Lambda(K.sum, arguments={'axis': 1})(_x2)

m1 = Concatenate()([BiLSTM1, _x1, Subtract()([BiLSTM1, _x1]), Multiply()([BiLSTM1, _x1])])
m2 = Concatenate()([BiLSTM2, _x2, Subtract()([BiLSTM2, _x2]), Multiply()([BiLSTM2, _x2])])

y1 = Bidirectional(LSTM(300, return_sequences=True))(m1)
y2 = Bidirectional(LSTM(300, return_sequences=True))(m2)

mx1 = Lambda(K.max, arguments={'axis': 1})(y1)
av1 = Lambda(K.mean, arguments={'axis': 1})(y1)
mx2 = Lambda(K.max, arguments={'axis': 1})(y2)
av2 = Lambda(K.mean, arguments={'axis': 1})(y2)

y = Concatenate()([av1, mx1, av2, mx2])
y = Dense(1024, activation='tanh')(y)
y = Dropout(0.5)(y)
y = Dense(1024, activation='tanh')(y)
y = Dropout(0.5)(y)
y = Dense(2, activation='softmax')(y)

model = Model(inputs=[inputs1, inputs2], outputs=y)
model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss='binary_crossentropy', metrics=['acc'])
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

history = model.fit([train_data1, train_data2], train_label, batch_size=512,
                    validation_data=([test_data1, test_data2], test_label), epochs=4, callbacks=callbacks_list, verbose=1)