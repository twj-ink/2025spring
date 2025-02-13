def find(x):
    if x!=parent[x]:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    if xp!=yp:
        parent[yp]=xp

while True:
    n,m=map(int,input().split())
    if {n,m}=={0}:
        break
    parent=[i for i in range(n)]
    for _ in range(m):
        s=list(map(int,input().split()))
        if len(s[1:])>=2:
            for i in range(2,len(s)):
                union(s[1],s[i])
    ans=sum(1 for i in range(n) if find(i)==find(0))
    print(ans)
