def find(x):
    if x!=parent[x]:
        parent[x]=find(parent[x])
    return parent[x]

def union(x,y):
    xp,yp=find(x),find(y)
    if xp==yp:
        print('Yes')
    else:
        parent[yp]=xp
        print('No')


while True:
    try:
        n,m=map(int,input().split())
        parent=[i for i in range(n+1)]
        for _ in range(m):
            a,b=map(int,input().split())
            union(a,b)
        cnt=[i for i in range(1,n+1) if i==parent[i]]
        print(len(cnt))
        print(*cnt)

    except EOFError:
        break