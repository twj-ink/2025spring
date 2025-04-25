class Vertex:
    def __init__(self, key:int):
        self.key = key
        self.neighbers = []

    def add_neighber(self, other):
        self.neighbers.append(other)

class Graph:
    def __init__(self):
        self.vertices = {}

    def add_edge(self, src, dest):
        if src not in self.vertices:
            Vsrc = Vertex(src)
            self.vertices[src] = Vsrc
        if dest not in self.vertices:
            Vdest = Vertex(dest)
            self.vertices[dest] = Vdest
        self.vertices[src].add_neighber(self.vertices[dest])

def construct(edges):
    g = Graph()
    for edge in edges:
        src, dest = edge
        g.add_edge(src, dest)
        g.add_edge(dest, src)

    ans = []
    for vertex in range(n):
        row = [0]*n
        if vertex in g.vertices:

            row[vertex] = len(g.vertices[vertex].neighbers)
            for neighber in g.vertices[vertex].neighbers:
                row[neighber.key] = -1
        ans.append(row)

    return ans


n,m=map(int,input().split())
edges=[]
for _ in range(m):
    a,b=map(int,input().split())
    edges.append((a,b))

ans=construct(edges)

for i in ans:
    print(*i)
