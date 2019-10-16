import json
from py2neo import Graph,Node,NodeMatcher,RelationshipMatcher

dbmovies = open('dbmovies','r',encoding='utf-8')
data = json.load(dbmovies)
#_ = [k for k in data[0]]
#print(_)
# ['id', 'title', 'url', 'cover', 'rate', 'director', 'composer', 'actor', 'category', 'district', 'language', 'showtime', 'length', 'othername']

class BuildGraph:
    def __init__(self):
        self.g = Graph(
            host="192.168.110.8",
            port=7687,
            user="neo4j",
            password="yiwiseneo")
        self.matcher = NodeMatcher(self.g)
        self.re_matcher = RelationshipMatcher(self.g)

    def get_node(self):
        movies = []  # id, title, url, cover, rate, showtime, length, othername
        person = []  # could be director  & composer & actor
        district = []
        language = []
        category = []
        for single in data:
            movie = {}
            movie['id'] = single['id']
            movie['title'] = single['title']
            movie['url'] = single['url']
            movie['cover'] = single['cover']
            movie['rate'] = single['rate']
            movie['showtime'] = single['showtime']
            movie['length'] = single['length']
            try:
                movie['othername'] = '、'.join(single['othername'])
            except:
                movie['othername'] = 'null'
            movies.append(movie)
            try:
                for _ in single['director']:
                    person.append(_.strip())
            except:
                pass
            try:
                for _ in single['composer']:
                    person.append(_)
            except:
                pass
            try:
                for _ in single['actor']:
                    person.append(_)
            except:
                pass
            try:
                for _ in single['district']:
                    district.append(_)
            except:
                pass
            try:
                for _ in single['language']:
                    language.append(_)
            except:
                pass
            try:
                for _ in single['category']:
                    category.append(_)
            except:
                pass
        return movies, person, district, language, category

    '''建立一般节点'''
    def create_node(self, label, nodes):
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
        return

    '''创建电影节点'''
    def create_movies_nodes(self, movies):
        for movie in movies:
            node = Node("Movie", id=movie['id'], name=movie['title'],
                        url=movie['url'] ,cover= movie['cover'],
                        rate=movie['rate'],showtime=movie['showtime'],
                        length=movie['length'],othername=movie['othername'])
            self.g.create(node)
        return

    def put_node(self, movies, person, district, language, category):
        person = set(person)
        district = set(district)
        language = set(language)
        category = set(category)
        self.create_movies_nodes(movies)
        self.create_node('Person', person)
        self.create_node('District', district)
        self.create_node('Language', language)
        self.create_node('Category', category)
        return

    def create_rela(self, start_node, end_node, p, q, rel_type):
        query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s]->(q)" % (
            start_node, end_node, p, q, rel_type)
        self.g.run(query)
        return

    def put_rela(self):
        for line in data:
            try:
                for tem in line['category']:
                    self.create_rela('Movie', 'Category', line['title'], tem, 'category_of')
            except:
                pass
                print(1,line)
            try:
                for tem in line['language']:
                    self.create_rela('Movie', 'Language', line['title'], tem, 'language_of')
            except:
                pass
                print(2,line)
            try:
                for tem in line['district']:
                    self.create_rela('Movie', 'District', line['title'], tem, 'district_of')
            except:
                pass
                print(3,line)
            try:
                for tem in line['actor']:
                    self.create_rela('Movie', 'Person', line['title'], tem, 'act_by')
            except:
                pass
                print(4,line)
            try:
                for tem in line['director']:
                    self.create_rela('Movie', 'Person', line['title'], tem, 'direct_by')
            except:
                pass
                print(5,line)
            try:
                for tem in line['composer']:
                    self.create_rela('Movie', 'Person', line['title'], tem, 'compose_by')
            except:
                pass
                print(6,line)



    def find_rate(self,name):
        res = []
        for i in self.matcher.match('Movie', name=name):
            res.append(i['rate'])
        return res
    def find_director(self,name):
        movie = self.matcher.match('Movie', name=name).first()
        res = []
        for i in self.re_matcher.match(nodes=(movie,), r_type='direct_by'):
            res.append(i.end_node['name'])
        return res
    def find_director_artwork(self, name):
        director = self.matcher.match('Person', name=name).first()
        all = self.re_matcher.match(nodes=(None, director), r_type='direct_by')
        res = []
        for i in all:
            res.append(i.start_node['name'])
        return res
    def find_actor(self,name):
        movie = self.matcher.match('Movie', name=name).first()
        res = []
        for i in self.re_matcher.match(nodes=(movie,), r_type='act_by'):
            res.append(i.end_node['name'])
        return res
    def find_actor_artwork(self, name):
        director = self.matcher.match('Person', name=name).first()
        all = self.re_matcher.match(nodes=(None, director), r_type='act_by')
        res = []
        for i in all:
            res.append(i.start_node['name'])
        return res
    def find_language(self,name):
        movie = self.matcher.match('Movie', name=name).first()
        res = []
        for i in self.re_matcher.match(nodes=(movie,), r_type='language_of'):
            res.append(i.end_node['name'])
        return res
    def find_category(self,name):
        movie = self.matcher.match('Movie', name=name).first()
        res = []
        for i in self.re_matcher.match(nodes=(movie,), r_type='category_of'):
            res.append(i.end_node['name'])
        return res
    def find_showtime(self,name):
        res = []
        for i in self.matcher.match('Movie', name=name):
            res.append(i['showtime'])
        return res




if __name__ == "__main__":
    handler = BuildGraph()
    movies, person, district, language, category = handler.get_node()
    handler.put_node(movies, person, district, language, category)
    handler.put_rela()


