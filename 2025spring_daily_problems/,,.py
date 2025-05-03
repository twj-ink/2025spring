# n=int(input())
# k=int(input())
# s=list(map(int,input().split()))
# dp=[[-float('inf')]*(k+1) for _ in range(n+1)]
# for i in range(k+1): dp[0][i]=0
# for i in range(1,n+1): dp[i][0]=0; dp[i][1]=s[i-1]
# for i in range(1,n+1):
#     for j in range(1,k+1):
#         for kk in range(1,i):
#             if dp[i-kk][j-1]!=-float('inf'):
#                 dp[i][j] = max(dp[i][j], s[kk-1]+dp[i-kk][j-1])
#
# print(dp[-1][-1])
def calculate_distinct_f_values(n):
    if n == 1:
        return 1  # 当 n = 1 时，只有一个排列，f(p) = 0

    # 初始化存储每个 n 的结果
    dp = [0] * (n + 1)
    dp[1] = 1  # 当 n = 1，只有1种可能，f(p) = 0

    # 递推方式（可以根据问题调整递推公式）
    for i in range(2, n + 1):
        # 遍历从1到n的所有排列组合，计算dp值
        dp[i] = dp[i - 1] * 2  # 这是简化版的递推关系，具体情况可以根据问题调整

    return dp[n]

# 测试
t = int(input())  # 读取测试用例数
for _ in range(t):
    n = int(input())  # 读取每个n
    print(calculate_distinct_f_values(n))  # 输出每个n对应的结果

