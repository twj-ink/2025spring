from functools import lru_cache

dx,dy=[0,-1,1,0],[-1,0,0,1]
@lru_cache(maxsize=1024)
def dfs(i,j,s,ans):
    for k in range(4):
        nx,ny=i+dx[k],j+dy[k]
        if 0<=nx<n and 0<=ny<m:




n,m=map(int,input().split())
s=[]
for _ in range(n):
    s.append(list(map(int,input().split())))

