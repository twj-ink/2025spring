def find(x):
    if x!=parent[x]:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    if xp==yp:
        return
    parent[xp]=yp

n,m=map(int,input().split())
parent=[i for i in range(n+1)]
for _ in range(m):
    z,x,y=map(int,input().split())
    if z==1:
        union(x,y)
    else:
        print(['N','Y'][find(x)==find(y)])