# 20250305 惊蛰 cs201 Mock Exam

### A ###
while True:
    try:
        s=input()
        cnt=s.count('@')
        if cnt==1:
            idx=s.find('@')
            if s[0] not in ('@','.') and s[-1] not in ('@','.') and \
                '.' in s[idx+2:] and s[idx-1]!='.' and s[idx+1]!='.':
                print('YES')
            else:
                print('NO')
        else:
            print('NO')
    except EOFError:
        break


### B ###
m=int(input())
s=list(input())
l=len(s)
n=l//m
a=[['.']*m for _ in range(n)]
# for i in a: print(*i)
for i in range(n):
    if not i%2==1:
        for j in range(m):
            a[i][j]=s[i*m+j]
    else:
        for j in range(m-1,-1,-1):
            a[i][j]=s[i*m+(m-1-j)]

# for i in a: print(*i)
ans=''
for j in range(m):
    for i in range(n):
        ans+=a[i][j]
print(ans)

#


### C ###
from collections import Counter
while True:
    n,m=map(int,input().split())
    if {n,m}=={0}:
        break
    s=[]
    for i in range(n):
        l=list(map(int,input().split()))
        s.extend(l)
    c=Counter(s)
    ans=[]
    cnts=[v for v in c.values()]
    cnts.sort(reverse=True)
    ans=[]
    for k,v in c.items():
        if v==cnts[1]:
            ans.append(k)
    ans.sort()
    print(*ans)






### D ###
d=int(input())
n=int(input())
s=[[0]*1025 for _ in range(1025)]
maxv=0
for _ in range(n):
    x,y,num=map(int,input().split())
    for i in range(max(0,x-d),min(1025,x+1+d)):
        for j in range(max(0,y-d),min(1025,y+1+d)):
            s[i][j]+=num
            maxv=max(maxv,s[i][j])
cnt=0
for i in range(1025):
    for j in range(1025):
        if s[i][j]==maxv:
            cnt+=1
print(cnt,maxv)





# ### E ###
dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]
def dfs(s,p,q,x,y,path):
    target=p*q
    s[x][y]=1
    xx=str(x+1)
    yy=chr(ord('A')+y)
    path+=(yy+xx)
    if len(path)==target*2:
        paths.append(path)
        return
    for i in range(8):
        nx,ny=x+dx[i],y+dy[i]
        if 0<=nx<p and 0<=ny<q and s[nx][ny]==0:
            s[nx][ny]=1
            dfs(s,p,q,nx,ny,path)
            s[nx][ny]=0

n=int(input())
for case in range(1,n+1):
    paths=[]
    p,q=map(int,input().split()) #1,2,3...,p;;;;a,b,c,,,,,,q
    s=[[0]*q for _ in range(p)]
    f=True
    for j in range(q):
        if not f:
            break
        for i in range(p):
            dfs(s,p,q,i,j,'')
            if paths:
                f=False
                break

    print(f'Scenario #{case}:')
    if not paths:
        print('impossible')
    else:
        paths.sort()
        print(paths[0])
    print()



### F ###
from heapq import heappop,heappush
def merge(a,b,n):
    heap,result,visited=[],[], {(0,0)}
    heappush(heap,(a[0]+b[0],0,0))
    while len(result)<n:
        res,i,j=heappop(heap)
        result.append(res)
        if i<n-1 and (i+1,j) not in visited:
            heappush(heap,(a[i+1]+b[j],i+1,j))
            visited.add((i+1,j))
        if j<n-1 and (i,j+1) not in visited:
            heappush(heap,(a[i]+b[j+1],i,j+1))
            visited.add((i,j+1))
    return result[:n]


t=int(input())
for _ in range(t):
    m,n=map(int,input().split())
    curr=sorted(list(map(int,input().split())))
    for _ in range(m-1):
        other=sorted(list(map(int,input().split())))
        curr=merge(curr,other,n)
    print(*curr)