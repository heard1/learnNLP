import numpy as np
import random
from tqdm import tqdm
import jieba
import re
import json
import keras
from keras.layers import *
from keras import Input
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

maxlen = 30
num_words = 10000
embedding_dim = 128
d_a = 128
inputs = Input(shape=(maxlen,))
embedding = Embedding(input_dim=num_words, input_length=maxlen, output_dim=embedding_dim, trainable=True)(inputs)
BiLSTM = Bidirectional(LSTM(units=256, return_sequences=True))(embedding)
dense = Dense(d_a, activation='tanh')(BiLSTM)
a = Dense(d_a, activation='softmax')(dense)
