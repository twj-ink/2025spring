#pylint:skip-file
dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]
def dfs(x,y,n,m,visited,step):
    global ans
    target=n*m
    if step==target:
        ans+=1
        return

    for i in range(8):
        nx,ny=x+dx[i],y+dy[i]
        if 0<=nx<n and 0<=ny<m and not visited[nx][ny]:
            visited[nx][ny]=True
            dfs(nx,ny,n,m,visited,step+1)
            visited[nx][ny]=False


t=int(input())
for _ in range(t):
    n,m,x,y=map(int,input().split())
    ans=0
    visited=[[False]*m for _ in range(n)]
    visited[x][y]=True
    dfs(x,y,n,m,visited,1)
    print(ans)