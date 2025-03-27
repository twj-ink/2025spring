def find(x):
    if parent[x]!=x:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    if xp!=yp:
        parent[xp]=yp
        size[yp]+=size[xp]

n,m=map(int,input().split())
parent=[i for i in range(n+1)]
size=[1]*(n+1)
for _ in range(m):
    a,b=map(int,input().split())
    union(a,b)
classes=[size[i] for i in range(1,n+1) if i==parent[i]]
print(len(classes))
classes.sort(reverse=True)
print(*classes)