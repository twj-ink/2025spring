from heapq import heappop,heapify

dx,dy=[0,-1,1,0],[-1,0,0,1]
n,m=map(int,input().split())
s=[]
heap=[]
for _ in range(n):
    s.append(list(map(int,input().split())))
for i in range(n):
    for j in range(m):
        heap.append((s[i][j],i,j))
heapify(heap)
dp=[[1]*m for _ in range(n)]
ans=1
while heap:
    height,x,y=heappop(heap)
    for i in range(4):
        nx,ny=x+dx[i],y+dy[i]
        if 0<=nx<n and 0<=ny<m and s[nx][ny]<height:
            dp[x][y]=max(dp[x][y],dp[nx][ny]+1)
    ans=max(ans,dp[x][y])
print(ans)
