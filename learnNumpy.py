import numpy as np
matrix = np.array([
['5', '10', '15'],
['5', '10', '15'],
['5', '10', '']
])
find = (matrix[:, 2] == '')
print(find)
matrix[find, 2] = '0'
print(matrix)

import jieba
sent = "我是练习时长两年半的个人练习生蔡徐坤，擅长唱跳rap篮球"
# 全模式，扫描所有词语
seg_list = jieba.cut(sent, cut_all=True)
# 精确模式
seg_list = jieba.cut(sent)
# 精确+全 搜索引擎模式
seg_list = jieba.cut_for_search(sent)
print('/'.join(seg_list))