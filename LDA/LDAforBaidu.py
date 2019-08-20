import jieba
from gensim import corpora, models
import json
import re

stop_words = ['呀','是','只','都','怎么样','可以','自己','咱们','就','吗','的','是不是','人','怎么','说','跟','不想','好','这方面','话','一定','肯定',
              '大','想','要','这','你','有没有','我','和','几个','了','找','下','这样','很','那','只有','能','什么样','你们','会','这个','有','一下','什么','没有','一般',
              '不会','呢','啥','还是','吧','的话','啊','请问','问','请','想问','怎么办','吗']

total = []
with open("res.json", encoding='utf-8') as f:
    for line in f.readlines():
        tem = json.loads(line)
        total.append(tem['question'])

regex = re.compile(u'[^\u4E00-\u9FA5|0-9a-zA-Z]')
def remove_punctuation(s):
    s_ = regex.sub('', s)
    return s_

texts = []
for line in total:
    tem = list(jieba.cut(remove_punctuation(line)))
    for i in tem:
        if i in stop_words:
            tem.remove(i)
    texts.append(tem)

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
documents = lda[corpus]

correct = []
error = []

for num, single in enumerate(list(documents)):
    if single[0][1] > single[1][1]:
        error.append(total[num])
    else:
        correct.append(total[num])
