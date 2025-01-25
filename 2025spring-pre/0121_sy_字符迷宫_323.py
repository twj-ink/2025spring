from collections import deque
dx,dy=[0,-1,1,0],[-1,0,0,1]
step=0
def bfs(sx,sy,ex,ey,s,n,m):
    global step
    q=deque()
    q.append((sx,sy))
    inq=set()
    inq.add((sx,sy))
    while q:
        for _ in range(len(q)):
            x,y=q.popleft()
            for i in range(4):
                nx,ny=x+dx[i],y+dy[i]
                if (nx,ny)==(ex,ey):
                    return step+1
                if 0<=nx<n and 0<=ny<m and s[nx][ny]=='.' and (nx,ny) not in inq:
                    q.append((nx,ny))
                    inq.add((nx,ny))
        step+=1
    return -1

n,m=map(int,input().split())
s=[]
for i in range(n):
    l=input()
    if 'S' in l:
        sx=i
        sy=l.index('S')
    if 'T' in l:
        ex=i
        ey=l.index('T')
    s.append(l)
print(bfs(sx,sy,ex,ey,s,n,m))
