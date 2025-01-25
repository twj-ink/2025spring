###题解使用两个nx和ny，第二个用
#nx,ny=x+dx[i]*2,y+dy[i]*2从而进行两次判断

###我使用这样的判断条件来避免出错
#if s[nx][ny]==s[x+dx[i]//2][y+dy[i]//2]==0:

# from collections import deque
# dx,dy=[0,0,-1,-2,1,2,0,0],[-1,-2,0,0,0,0,1,2]
# step=0
# def bfs(sx,sy,ex,ey,s,n,m):
#     global step
#     q=deque()
#     q.append((sx,sy))
#     inq=set()
#     inq.add((sx,sy))
#     while q:
#         for _ in range(len(q)):
#             x,y=q.popleft()
#             for i in range(8):
#                 nx,ny=x+dx[i],y+dy[i]
#                 if 0<=nx<n and 0<=ny<m and s[x+dx[i]//2][y+dy[i]//2]==0 and (nx,ny)==(ex,ey):
#                     return step+1
#                 if 0<=nx<n and 0<=ny<m and s[nx][ny]==s[x+dx[i]//2][y+dy[i]//2]==0 and (nx,ny) not in inq:
#                     q.append((nx,ny))
#                     inq.add((nx,ny))
#         step+=1
#     return -1

# n,m=map(int,input().split())
# # s=[]
# # for i in range(n):
# #     l=input()
# #     if 'S' in l:
# #         sx=i
# #         sy=l.index('S')
# #     if 'T' in l:
# #         ex=i
# #         ey=l.index('T')
# #     s.append(l)
# s=[[int(i) for i in input().split()] for _ in range(n)]
# print(bfs(0,0,n-1,m-1,s,n,m))
from collections import deque

# 四个基本方向：上、下、左、右
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]


# 判断坐标是否在迷宫的范围内
def in_bounds(x, y, n, m):
    return 0 <= x < n and 0 <= y < m


# 广度优先搜索 (BFS) 计算最小步数
def bfs(maze, n, m):
    # 初始化步数矩阵，-1表示未访问
    steps = [[-1] * m for _ in range(n)]
    steps[0][0] = 0  # 起点的步数为0

    # BFS队列初始化
    queue = deque([(0, 0)])

    # 开始BFS
    while queue:
        x, y = queue.popleft()

        # 如果到达终点，返回步数
        if x == n - 1 and y == m - 1:
            return steps[x][y]

        # 遍历四个基本方向
        for i in range(4):
            # 一格移动
            next_x1, next_y1 = x + dx[i], y + dy[i]
            # 两格移动
            next_x2, next_y2 = x + 2 * dx[i], y + 2 * dy[i]

            # 检查是否可以移动一格
            if in_bounds(next_x1, next_y1, n, m) and maze[next_x1][next_y1] == 0 and steps[next_x1][next_y1] == -1:
                steps[next_x1][next_y1] = steps[x][y] + 1
                queue.append((next_x1, next_y1))

            # 检查是否可以移动两格
            if in_bounds(next_x2, next_y2, n, m) and maze[next_x1][next_y1] == 0 and maze[next_x2][next_y2] == 0 and \
                    steps[next_x2][next_y2] == -1:
                steps[next_x2][next_y2] = steps[x][y] + 1
                queue.append((next_x2, next_y2))

    # 如果无法到达终点，返回-1
    return -1


# 读取输入
n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

# 执行BFS，计算最小步数
result = bfs(maze, n, m)
print(result)