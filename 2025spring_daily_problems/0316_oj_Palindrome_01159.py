# 超内存

'''
n=int(input())
s=input()
dp=[[0]*n for _ in range(n)]
for j in range(n):
    for i in range(j,-1,-1):
        if i>=j: continue
        if s[i]==s[j]:
            dp[i][j]=dp[i+1][j-1]
        else:
            dp[i][j]=min(dp[i+1][j]+1, dp[i][j-1]+1, dp[i+1][j-1]+2)

print(dp[0][-1])
'''

# curr存的是dp[0..n][j]这一当前列，
# prev存的是dp[0..n][j-1]这一先前列，
# 每次用到的就只有当前格子的左边、下边、左下边三个位置，即
# dp[i][j-1]   ->  prev[i]
# dp[i+1][j]   ->  curr[i+1]
# dp[i+1][j-1] ->  prev[i+1]

#也就是说：取j的时候就是curr，取j-1的时候就是prev，i的值保持不变

n = int(input())
s = input()

prev = [0] * n  # 存储上一行 dp[i+1][j]
curr = [0] * n  # 存储当前行 dp[i][j]

for j in range(1, n):  # 右端点 j 从 1 到 n-1
    for i in range(j - 1, -1, -1):  # 左端点 i 逆序遍历
        if s[i] == s[j]:
            curr[i] = prev[i + 1] if i + 1 <= j - 1 else 0
        else:
            curr[i] = min(prev[i] + 1, curr[i + 1] + 1, prev[i + 1] + 2 if i + 1 <= j - 1 else float('inf'))

    prev, curr = curr, prev  # 滚动数组，交换 prev 和 curr

print(prev[0])  # 最终答案存储在 prev[0]
