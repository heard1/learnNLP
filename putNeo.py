from py2neo import Graph,Node,Relationship
from py2neo.matching import NodeMatcher
import json

json_file = open('./microeco.json', encoding='utf-8')
f = json.load(json_file)

g = Graph(
            host="192.168.110.8",
            port=6687,
            user="neo4j",
            password="yiwiseneo")
matcher = NodeMatcher(g)

for line in f:
    from_node = matcher.match('obj', name=line['object']).first()
    if from_node is None:
        from_node = Node('obj', name=line['object'])
        g.create(from_node)
    to_node = matcher.match('obj', name=line['subject']).first()
    if to_node is None:
        to_node = Node('obj', name=line['subject'])
        g.create(to_node)
    rela = Relationship(from_node, line['relation'], to_node)
    g.create(rela)