from py2neo import Graph,Node,NodeMatcher,RelationshipMatcher
import json

class NeoGraph:
    def __init__(self):
        self.g = Graph(
            host="10.15.82.71",
            port=57687,
            user="neo4j",
            password="123")
        self.matcher = NodeMatcher(self.g)
        self.re_matcher = RelationshipMatcher(self.g)

    def getNode(self, key):
        return self.matcher.match(name = key).first()

    def getBansui(self, key):
        node = self.getNode(key)
        res = []
        if node is None:
            return res
        for i in self.re_matcher.match(nodes=(None,node), r_type='伴随'):
            res.append([len(i['id']), i.start_node['name']])
            
        for i in self.re_matcher.match(nodes=(node,None), r_type='伴随'):
            res.append([len(i['id']), i.end_node['name']])
        return res

    def getZhiliao(self, key):
        node = self.getNode(key)
        res = []
        if node is None:
            return res
        for i in self.re_matcher.match(nodes=(None,node), r_type='治疗'):
            res.append([len(i['id']), i.start_node['name']])
            
        for i in self.re_matcher.match(nodes=(node,None), r_type='治疗'):
            res.append([len(i['id']), i.end_node['name']])
        return res

    def getZuchen(self, key):
        node = self.getNode(key)
        res = []
        if node is None:
            return res
        for i in self.re_matcher.match(nodes=(None,node), r_type='组成'):
            res.append([len(i['id']), i.start_node['name']])
            
        for i in self.re_matcher.match(nodes=(node,None), r_type='组成'):
            res.append([len(i['id']), i.end_node['name']])
        return res    
    
handler = NeoGraph()

confidence = {1:"可信度5%", 
              2:"可信度10%",
              3:"可信度15%",
              4:"可信度30%",
              5:"可信度40%",
              6:"可信度50%",
              7:"可信度60%"}

def getJson(word):
    res = {}
    bansui = handler.getBansui(word)
    zhiliao = handler.getZhiliao(word)
    zuchen = handler.getZuchen(word)
    if len(bansui)>10:
        bansui = sorted(bansui, key=lambda x:x[0])[-8:]
    if len(zhiliao)>10:
        zhiliao = sorted(zhiliao, key=lambda x:x[0])[-8:]
    if len(zuchen)>10:
        zuchen = sorted(zuchen, key=lambda x:x[0])[-8:]
    for i in bansui:
        i[0] = "伴随：" + confidence.get(i[0],"可信度90%")
    for i in zhiliao:
        i[0] = "治疗：" + confidence.get(i[0],"可信度90%")
    for i in zuchen:
        i[0] = "组成：" + confidence.get(i[0],"可信度90%")
    bansui.extend(zhiliao)
    bansui.extend(zuchen)
    data = {}
    data['entity'] = word
    data['avp'] = bansui
    data['tag'] = str(handler.getNode(word).labels)
    res['message'] = "success"
    res['data'] = data
    return res

from flask import Flask, request
from flask_cors import CORS,cross_origin

def create_app():
    app = Flask(__name__)
    @app.route('/', methods=['GET', 'POST'])
    @cross_origin()
    def callback():
        entity = request.args.get("entity") or "EOF"
        try:
            res = getJson(entity)
            return json.dumps(res, ensure_ascii=False, indent=4)
        except:
            return json.dumps({'message':'error'})
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5787)