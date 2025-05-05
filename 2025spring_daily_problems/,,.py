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
for _ in range(int(input())):
    n, k = map(int, input().split())
    if k & 1:
        print('No')
        continue

    max_v = n ** 2 // 2
    if k > max_v:
        print('No')
        continue

    p = [i for i in range(1, n + 1)]
    i, j = 0, n - 1
    while i < j:
        if k >= (curr := 2 * (p[j] - p[i])):
            k -= curr
            p[i], p[j] = p[j], p[i]
            i += 1
            j -= 1
        else:
            j -= 1
    if k == 0:
        print('Yes')
        print(*p)
    else:
        print('No')
