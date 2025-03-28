# dp[i][j]：表示当前还有i个待入的元素，并且栈中还剩 j 个元素的情况下的合法出栈序列数。
# 状态转移：
# 如果i=0，都在栈里，直接为1
# 如果j=0，无法弹出，dp[i][j]=dp[i-1][j+1]
# 如果j!=0，dp[i][j]=dp[i-1][j+1]+dp[i][j-1]
import sys
sys.setrecursionlimit(1<<30)
def dfs(i, j):
    if i == 0:
        return 1
    if j == 0:
        return dfs(i - 1, j + 1)  # append stack
    if j == n:
        return dfs(i, j - 1)  # pop
    return dfs(i - 1, j + 1) + dfs(i, j - 1)


n = int(input())
print(dfs(n, 0))