# 1 西 2 北 4东 8
# 3=12 5=14 9=18 6=24 10=28 12=48 7=124 11=128
from collections import deque
n=int(input())
m=int(input())
s=[]
for _ in range(n):
    s.append(list(map(int,input().split())))

w=[[[0,0,0,0] for _ in range(m)] for _ in range(n)]
d={
    0:[0,0,0,0],
    1:[1,0,0,0], 2:[0,1,0,0], 4:[0,0,1,0], 8:[0,0,0,1],
    3:[1,1,0,0], 5:[1,0,1,0], 9:[1,0,0,1], 6:[0,1,1,0], 10:[0,1,0,1], 12:[0,0,1,1],
    7:[1,1,1,0], 11:[1,1,0,1], 13:[1,0,1,1], 14:[0,1,1,1],
    15:[1,1,1,1],
}
for i in range(n):
    for j in range(m):
        w[i][j]=d[s[i][j]]

dir=[(0,-1),(-1,0),(0,1),(1,0)]
cnt=0
maxS=0
inq=set()
def bfs(x,y,w,curr):
    global inq,cnt,maxS
    q=deque()
    q.append((x,y))
    inq.add((x,y))
    while q:
        x,y=q.popleft()
        for i in range(4):
            if w[x][y][i]!=1:
                dx,dy=dir[i]
                nx,ny=x+dx,y+dy
                if 0<=nx<n and 0<=ny<m and (nx,ny) not in inq:
                    q.append((nx,ny))
                    inq.add((nx,ny))
                    curr+=1
    cnt+=1
    maxS=max(maxS,curr+1)


for i in range(n):
    for j in range(m):
        if (i,j) not in inq:
            bfs(i,j,w,0)

print(cnt)
print(maxS)