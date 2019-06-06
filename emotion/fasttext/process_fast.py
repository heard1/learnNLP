# import fasttext
# model=fasttext.skipgram('train.csv','ski')
# # model = fasttext.load_model('ski.bin')
# print(model['你好'])
#
# c = fasttext.supervised('train.csv','model',label_prefix='label')
# re=c.test('valid.csv')
# print(re.precision) #86%
# sgd 72
# svc 82
# rf 83
# gbdt 80
from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']
# define class labels
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
