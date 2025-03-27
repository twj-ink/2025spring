def find(x):
    if x!=parent[x]:
        parent[x]=find(parent[x])
    return parent[x]

def union(a,b):
    ap,bp=find(a),find(b)
    if ap!=bp:
        parent[ap]=bp

case=0
while True:
    n,m=map(int,input().split())
    if {n,m}=={0}:
        break
    case+=1
    parent=[i for i in range(n+1)]
    for _ in range(m):
        a,b=map(int,input().split())
        union(a,b)
    ans=sum(1 for i in range(1,n+1) if i==parent[i])
    print(f'Case {case}: {ans}')