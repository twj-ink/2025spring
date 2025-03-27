#每出现一个数字就将其与本身+1链接

def find(x):
    if x!=parent[x]:
        parent[x]=find(parent[x])
    return parent[x]

def union(x):
    parent[find(x)]=find(x)+1

n=int(input())
s=list(map(int,input().split()))
parent=[i for i in range(n+1)]
ans=[]
for i in range(n):
    ans.append(find(s[i]))
    union(s[i])
print(*ans)