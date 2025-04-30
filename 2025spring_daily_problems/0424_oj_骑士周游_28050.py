dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]

def can_move(n,s,x,y):
    return 0<=x<n and 0<=y<n and s[x][y]==0

def get_degree(x,y):
    cnt=0
    for i in range(8):
        nx,ny=x+dx[i],y+dy[i]
        if can_move(n,s,nx,ny):
            cnt+=1
    return cnt

def dfs(n,r,c,step,s,sr,sc):
    if step==n**2:
        return True
    degrees=[]
    for i in range(8):
        nx,ny=r+dx[i],c+dy[i]
        if (can_move(n,s,nx,ny) or (step==n**2-1 and (nx,ny)==(sr,sc))):
            deg=get_degree(nx,ny)
            degrees.append((deg,nx,ny))

    degrees.sort()
    for _, nx, ny in degrees:
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