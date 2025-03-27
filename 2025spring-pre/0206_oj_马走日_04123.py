#pylint:skip-file
dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]

def dfs(visited,n,m,x,y,cell,cnt):
    target=n*m
    if cell==target:
        cnt[0]+=1
        return
    for i in range(8):
        nx,ny=x+dx[i],y+dy[i]
        if 0<=nx<n and 0<=ny<m and not visited[nx][ny]:
            cell+=1
            visited[nx][ny]=True
            dfs(visited,n,m,nx,ny,cell,cnt)
            cell-=1
            visited[nx][ny]=False


for _ in range(int(input())):
    n,m,x,y=map(int,input().split())
    # s=[[0]*m for _ in range(n)]
    visited=[[False]*m for _ in range(n)]
    visited[x][y]=True
    cnt=[0]
    dfs(visited,n,m,x,y,1,cnt)
    print(cnt[0])