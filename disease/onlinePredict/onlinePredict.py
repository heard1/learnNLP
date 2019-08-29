from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from jieba import posseg
import re


def remove_punctuation(s):
    """
    文本标准化：仅保留中文字符、英文字符和数字字符
    :param s: 输入文本（中文文本要求进行分词处理）
    :return: s_：标准化的文本
    """
    regex = re.compile(u'[^\u4E00-\u9FA5]')
    s_ = regex.sub('', s)
    return s_

def get_stop_word():
    stop = []
    with open('stopWord.txt',encoding='utf-8') as f:
        for line in f.readlines():
            stop.append(line[:-1])
    return stop

def getCor(new_sentence):

    stop = get_stop_word()
    res = []

    new = remove_punctuation(new_sentence)
    stringword = ''
    for word in jieba.cut(new):
        if word not in stop:
            stringword = stringword + ' ' + word
    res.append(stringword)

    with open('cor.txt', encoding='utf-8') as f:
        for single in f.readlines():

            num, sen = single.split('\t')
            sen = remove_punctuation(sen)
            stringword = ''
            for word in jieba.cut(sen):
                if word not in stop:
                    stringword = stringword+' '+word
            res.append(stringword)
    return res


def get_tfidf_socre(new):
    corpus = getCor(new)
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    weight = tfidf.toarray()

    dict_one={}
    for i in range(1,5):
        if (weight[0][weight[0].argsort()[-i]]) > 0:
            dict_one[word[weight[0].argsort()[-i]]] = weight[0][weight[0].argsort()[-i]]

    return dict_one

def get_pos_score(word,ori):
    w = posseg.cut(word)
    pos = ''
    score = 1
    for i in w:
        pos = i.flag
        if pos[0] == 'n':
            score = 2
        if pos[0] == 'v':
            score = 1.5
        if pos[0] == 'a':
            score = 1.2
    return score*ori

def get_common_word():
    common = []
    with open('symptom.txt',encoding='utf-8') as f:
        for line in f.readlines():
            common.append(line[:-1])
    with open('plant.txt',encoding='utf-8') as f:
        for line in f.readlines():
            common.append(line[:-1])
    with open('disease.txt',encoding='utf-8') as f:
        for line in f.readlines():
            common.append(line[:-1])
    common.remove('李')
    return common




if __name__ == '__main__':
    while True:
        new = input()
        tot = get_tfidf_socre(new)
        common = get_common_word()
        tot2 = new

        tem_dic = {}
        for word in tot:
            tem_dic[word] = get_pos_score(word, tot[word])
        for i in common:
            if i in tot2:
                tem_dic[i] = 5
        print(tem_dic)