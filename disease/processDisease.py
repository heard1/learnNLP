import jieba
import re

'''
regex = re.compile(u'[^\u4E00-\u9FA5]')
def remove_punctuation(s):
    """
    文本标准化：仅保留中文字符、英文字符和数字字符
    :param s: 输入文本（中文文本要求进行分词处理）
    :return: s_：标准化的文本
    """
    s_ = regex.sub('', s)
    return s_

tot = {}
count = 0
with open('qa.txt', encoding='utf-8') as f:
    for single in f.readlines():
        count += 1
        if count % 1000 == 0:
            print(count)
        num, sen = single.split('\t')
        sen = remove_punctuation(sen)
        for word in jieba.cut(sen):
            if tot.get(word) is None:
                tot[word] = 1
            else:
                tot[word] = tot[word] + 1



f = open('stop_word.txt','w',encoding='utf-8')
for single in sorted(tot.items(),key=lambda d:d[1], reverse = True):
    f.write(single[0]+'\n')
f.close()

par = re.compile(u'[0-9a-zA-Z]')
new = open('stop_word2.txt','w',encoding='utf-8')
f = open('stop_word.txt',encoding='utf-8')
for single in f.readlines():
    if par.match(single) is None:
        new.write(single)
'''