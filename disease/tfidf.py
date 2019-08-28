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

def getCor():
    count = 0
    stop = get_stop_word()
    res = []
    with open('qa.txt', encoding='utf-8') as f:
        for single in f.readlines():
            count += 1
            if count % 1000 == 0:
                print(count)
            num, sen = single.split('\t')
            sen = remove_punctuation(sen)
            stringword = ''
            for word in jieba.cut(sen):
                if word not in stop:
                    stringword = stringword+' '+word
            res.append(stringword)
    return res


def get_tfidf_socre():
    corpus = getCor()
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    tot = []
    for i in range(707):
        print(i)
        # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus[i*1000:(i+1)*1000]))

        # 获取词袋模型中的所有词语
        word = vectorizer.get_feature_names()
        weight = tfidf.toarray()

        for single in weight:
            dict_one={}
            for i in range(1,5):
                if (single[single.argsort()[-i]]) > 0:
                    dict_one[word[single.argsort()[-i]]] = single[single.argsort()[-i]]
            tot.append(dict_one)
    return tot

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

'''
if __name__ == '__main__':
    tot = get_tfidf_socre()
    common = get_common_word()
    tot2 = []
    #final = []
    res_tag = open('store.txt','a+',encoding='utf-8')
    count = 0
    with open('qa.txt', encoding='utf-8') as f:
        for single in f.readlines():
            num, sen = single.split('\t')
            tot2.append(sen)
    for question,question2 in zip(tot,tot2):
        count += 1
        if count % 1000 == 0:
            print(count)
        tem_dic={}
        for word in question:
            tem_dic[word] = get_pos_score(word, question[word])
        for i in common:
            if i in question2:
                tem_dic[i] = 5
        #final.append(tem_dic)
        res_tag.write(str(tem_dic)+'\n')
    res_tag.close()
'''