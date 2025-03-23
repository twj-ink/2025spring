from collections import deque
x,y=map(int,input().split())
sx,sy=x,y
ex,ey=map(int,input().split())
m=int(input())
s=[[0]*11 for _ in range(11)]
ss=[[list() for _ in range(11)] for _ in range(11)]
for _ in range(m):
    i,j=map(int,input().split())
    s[i][j]=1

cnt=0
paths=[]
dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]
def canmove(x,y,dx,dy):
    return s[x+dx//2][y+dy//2]!=1
def getpath(x,y,curr):
    # print(ss[x][y])
    # print(curr)
    if curr and curr[-1]==(sx,sy):
        return curr[::-1]
    curr.append((x,y))
    nx,ny=ss[x][y][0]
    # print((nx,ny))
    return getpath(nx,ny,curr)

def bfs(x,y,ex,ey):
    global cnt,paths
    q=deque()
    q.append((x,y))
    inq=set()
    inq.add((x,y,-1,-1))
    while q:
        for _ in range(len(q)):
            x,y=q.popleft()
            for i in range(8):
                d1,d2=dx[i],dy[i]
                if canmove(x,y,d1,d2):
                    nx,ny=x+d1,y+d2
                    if 0<=nx<11 and 0<=ny<11 and (nx,ny,x,y) not in inq and s[nx][ny]==0:
                        q.append((nx,ny))
                        inq.add((nx,ny,x,y))
                        ss[nx][ny].append((x,y))
                        # for i in ss:print(*i)
                        # print()
        if len(ss[ex][ey])>1:
            print(len(ss[ex][ey]))
            return
        elif len(ss[ex][ey])==1:
            path=getpath(ex,ey,[])
            path='-'.join(map(str,path))
            for i in path:
                print(i.strip(),end='')
            return

bfs(x,y,ex,ey)
