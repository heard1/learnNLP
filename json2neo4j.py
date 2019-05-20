from py2neo import Graph, Node, Relationship
import pandas as pd

graph = Graph("http://localhost:7474", username="graph", password='123456')
with open("baike.json", encoding='utf-8') as f:
    data = f.read()
    data = data.replace('\n','')
    data = eval(data)
    data = [single['objection'] for single in data]

'''
# 提取所有实体 data[i]['key']，用set防止重复
entity = set()
for tot in data:
    for key in tot.keys():
        if key != 'description':
            entity.add(tot[key])
for x in entity:
    tem = Node(x)
    graph.create(tem)
# 部分实体加入description

# 提取所有关系


for tot in data:
    obj = Node(tot['obj'])
    for key in tot.keys():
        if key != 'obj' and key != 'description':
            tem = Node(tot[key])
            r = Relationship(obj, key, tem)
            s = obj | tem | r
            graph.create(s)

'''
obj = []
desc = []
for tot in data:
    obj.append('\"'+tot['obj']+'\"')
    desc.append('\"'+tot['description']+'\"')

dataframe = pd.DataFrame({obj: obj, desc: desc})
dataframe.to_csv("test.csv", index=False, sep=',')
