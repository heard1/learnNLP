from gensim import corpora

texts = [['human', 'interface', 'computer'],
['survey', 'user', 'computer', 'system', 'response', 'time'],
['eps', 'user', 'interface', 'system'],
['system', 'human', 'system', 'eps'],
['user', 'response', 'time'],
['trees'],
['graph', 'trees'],
['graph', 'minors', 'trees'],
['graph', 'minors', 'survey']]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

#TFIDF词向量
from gensim import models
tfidf = models.TfidfModel(corpus)
single_line = dictionary.doc2bow(['human', 'interface', 'computer'])
single_line = tfidf[single_line]

# 聚类
lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
documents = lsi_model[corpus]
query = dictionary.doc2bow(['human', 'interface', 'computer'])
query_vec = lsi_model[query]

lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2)
documents2 = lda[corpus]
query = dictionary.doc2bow(['human', 'interface', 'computer'])
query_vec2 = lda[query]
