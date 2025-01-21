#pylint:skip-file
def dfs(i,k,col,curr):
    global cnt
    if curr==k:
        cnt+=1
        return
    if i==n or n-i+curr<k:#剪枝
        return

    for j in range(n):
        if not col[j] and s[i][j]=='#':
            curr+=1
            col[j]=True
            dfs(i+1,k,col,curr) #在井号放置棋子
            curr-=1
            col[j]=False
    #当j列遍历结束后，这一行肯定是没有放置棋子的
    #接下来要进入下一行，继续查看能不能放棋子
    dfs(i+1,k,col,curr) #下一层

while True:
    n,k=map(int,input().split())
    if n==k==-1:
        break
    s=[]
    for _ in range(n):
        s.append(input())

    cnt=0
    dfs(0,k,[False]*n,0)
    print(cnt)

#################################################
#八皇后
# def solve_n_queens(n):
#     def dfs(row):
#         # 如果所有皇后都成功放置
#         if row == n:
#             result.append(board[:])
#             return
#
#         for col in range(n):
#             # 检查是否可以放置皇后
#             if col in cols or row - col in diag1 or row + col in diag2:
#                 continue
#
#             # 放置皇后
#             cols.add(col)
#             diag1.add(row - col)
#             diag2.add(row + col)
#             board[row] = col
#
#             # 递归处理下一行
#             dfs(row + 1)
#
#             # 撤销放置
#             cols.remove(col)
#             diag1.remove(row - col)
#             diag2.remove(row + col)
#             board[row] = -1
#
#             #此处必定会产生合法解，因此无须跳过这一行，因为每一行都要放置
#
#     result = []
#     cols = set()  # 列是否被占用
#     diag1 = set()  # 主对角线是否被占用
#     diag2 = set()  # 副对角线是否被占用
#     board = [-1] * n  # 每行皇后的位置
#     dfs(0)
#     return result
#
# # 示例：求解 8 皇后问题
# n = 8
# solutions = solve_n_queens(n)
# print(f"Number of solutions for {n}-queens: {len(solutions)}")
