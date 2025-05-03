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
    n,x=map(int,input().split())
    c=sum(1 for i in bin(x)[2:] if i=='1')
    print(bin(x)[2:])
    if c==0:
        if n==1:
            print(-1)
        elif n%2==0:
            print(n)
        else:
            print(n+3)
    elif c==1:
        if n%2==0:
            print(n+3)
        else:
            print(n)
    else:
        print(x + (max(0, n - c) + 1) // 2 * 2)