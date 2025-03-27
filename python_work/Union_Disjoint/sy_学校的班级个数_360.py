def find(x):
    if parent[x]!=x:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    parent[xp]=yp



n,m=map(int,input().split())
# parent=[int(i) for i in range(n+1)]
parent=list(range(n+1))
for _ in range(m):
    a,b=map(int,input().split())
    union(a,b)

# classes=set(find(i) for i in range(1,n+1))
# print(len(classes))
print(sum(1 for i in range(1,n+1) if i==parent[i]))