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

'''
cnt not -1 == 1 -> if min(a)+k<s -> 0 else 1
not -1 > 1 -> sum of dif -> 0
              sum of s  if upper(s)!=n -> 0


not -1 == 0 -> k-(dif(S)-1)
'''


def can_satisfy(a, b, k):
    n = len(a)
    m = len(b)

    # 枚举插入位置
    for pos in range(n + 1):
        new_a = a[:pos] + [k] + a[pos:]
        i = 0  # new_a index
        j = 0  # b index
        while i < len(new_a) and j < m:
            if new_a[i] >= b[j]:
                j += 1
            i += 1
        if j == m:
            return True
    return False


def solve_case(n, m, a, b):
    # 先判断不插花是否满足
    i = 0
    j = 0
    while i < n and j < m:
        if a[i] >= b[j]:
            j += 1
        i += 1
    if j == m:
        return 0

    # 二分查找最小k
    left = 1
    right = int(1e9)
    answer = -1
    while left <= right:
        mid = (left + right) // 2
        if can_satisfy(a, b, mid):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    return answer


# 主程序
t = int(input())
results = []
for _ in range(t):
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    results.append(solve_case(n, m, a, b))

# 输出结果
for res in results:
    print(res)
