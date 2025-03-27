from collections import deque
dx,dy=[0,-1,1,0],[-1,0,0,1]
step=0
def bfs(x,y,n,m):
    global step
    q=deque()
    q.append((x,y))
    inq=set()
    inq.add((x,y))
    while q:
        for _ in range(len(q)):
            x,y=q.popleft()
            for i in range(4):
                nx,ny=x+dx[i],y+dy[i]
                if (nx,ny)==(n-1,m-1):
                    step+=1
                    return step
                if 0<=nx<n and 0<=ny<m and s[nx][ny]==0 and (nx,ny) not in inq:
                    q.append((nx,ny))
                    inq.add((nx,ny))
        step+=1
    return -1

n,m=map(int,input().split())
s=[[int(i) for i in input().split()] for _ in range(n)]
print(bfs(0,0,n,m))