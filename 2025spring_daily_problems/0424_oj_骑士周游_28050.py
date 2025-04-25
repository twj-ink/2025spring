dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]

def dfs(n,r,c,step,s,sr,sc):
    if step==n**2:
        return True
    for i in range(8):
        nx,ny=r+dx[i],c+dy[i]
        if ((0<=nx<n and 0<=ny<n and s[nx][ny]==0) or (step==n**2-1 and (nx,ny)==(sr,sc))):
            s[nx][ny]=1
            if dfs(n,nx,ny,step+1,s,sr,sc):
                return True
            s[nx][ny]=0
    return False

n=int(input())
r,c=map(int,input().split())
s=[[0]*n for _ in range(n)]
s[r][c]=1

print(['fail','success'][dfs(n,r,c,1,s,r,c)])