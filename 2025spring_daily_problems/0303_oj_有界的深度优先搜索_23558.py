from collections import defaultdict

def dls(ans,start,d1,l,step,visited):
    if step==l:
        return
    for node in d1[start]:
        if node not in visited:
            ans.append(node)
            visited.add(node)
            dls(ans,node,d1,l,step+1,visited)

n,m,l=map(int,input().split())
d1=defaultdict(list)
for _ in range(m):
    a,b=map(int,input().split())
    d1[a].append(b)
    d1[b].append(a)
for v in d1.values():
    v.sort()

start=int(input())
ans=[start]
dls(ans,start,d1,l,0,{start})
print(*ans)