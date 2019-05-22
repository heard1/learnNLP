from py2neo import Graph, Node, Relationship, NodeMatcher, Subgraph
# connect
graph = Graph("http://localhost:7474", username="graph", password='123456')
# create node and relationship
a = Node("Person", name="Alice", gender="male")
b = Node("Person", name="Bob")
relationab = Relationship(a, "KNOWS", b)
graph.create(relationab)

# change node and relationship
a['name'] = None
a['name'] = "Aliceee"

print(len(a), a.labels, dict(a))

mydict = {"tem":"qwq"}
a.update(mydict)

# find Nodes
tx = graph.begin()
matcher = NodeMatcher(graph)
nodes = matcher.match("Person")
listNodes = list(nodes)

# change Nodes
for node in listNodes:
    node['tem'] = 666
sub = Subgraph(nodes=listNodes)
tx.push(sub)
tx.commit()