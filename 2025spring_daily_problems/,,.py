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
from math import ceil
def success(mid,n):
    cnt=ceil(n*0.6)
    a=mid/1000000000
    for i in range(cnt):
        converted=a*s[i]+1.1**(a*s[i])
        if converted<85:
            return False
    return True

s=list(map(float,input().split()))
n=len(s)
s.sort(reverse=True)
left,right=0,1000000000
while left<=right:
    mid=(left+right)//2
    if success(mid,n):
        right=mid-1
    else:
        left=mid+1
print(left)