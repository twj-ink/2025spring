def find(x):
    if parent[x]!=x:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    parent[xp]=yp

n,m=map(int,input().split())
parent=[i for i in range(n+1)]
for _ in range(m):
    a,b=map(int,input().split())
    union(a,b)
p=int(input())
for _ in range(p):
    x,y=map(int,input().split())
    if find(x)==find(y):
        print('Yes')
    else:
        print('No')