import sys
import os
import pandas as pd
import numpy as np
import random
import json

io = pd.io.excel.ExcelFile('liepin_04.xlsx')
data = pd.read_excel(io,sheet_name=u'intent_text',encoding='utf-8',sep='\t')
data.dropna(how='any', inplace=True)
data.drop_duplicates(inplace=True)

all_dict_result = {}
for i,row in enumerate(data.values):
    intent = row[0]
    text = row[1]
    all_dict_result.setdefault(intent, []).append(text)

require_size = 0

to_expand_data = {}
use_data = {}
for intent in all_dict_result:
    if len(all_dict_result[intent]) < require_size:
        to_expand_data[intent] = all_dict_result[intent]
    else:
        use_data[intent] = all_dict_result[intent]

for intent in all_dict_result:
    for i in range(len(all_dict_result[intent])):
        if isinstance(all_dict_result[intent][i],int):
            all_dict_result[intent][i] = str(all_dict_result[intent][i])

all_dict_result_copy = {}
for intent in all_dict_result:
    all_dict_result_copy[intent] = all_dict_result[intent]

data_dict_checked = {}

for intent in all_dict_result_copy:
    text_list = all_dict_result_copy[intent]
    for text in text_list:
        data_dict_checked.setdefault(intent,[]).append(text)

with open('liepin_data.json',mode='w',encoding='utf-8') as f:
    json.dump(data_dict_checked,f,ensure_ascii=False,indent=4)
