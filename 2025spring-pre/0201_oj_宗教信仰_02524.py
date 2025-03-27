def find(x):
    if x!=parent[x]:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    if xp!=yp:
        # if rank[xp]>=rank[yp]:
            parent[yp]=xp
            # rank[xp]+=1
        # else:
        #     parent[xp]=yp
        #     rank[yp]+=1


cnt=0
while True:
    cnt+=1
    n,m=map(int,input().split())
    if n==m==0:
        break
    parent=[i for i in range(n+1)]
    rank=[1]*(n+1)
    for _ in range(m):
        a,b=map(int,input().split())
        union(a,b)
    ans=[i for i in range(1,n+1) if i==parent[i]]
    print(f'Case {cnt}: {len(ans)}')

