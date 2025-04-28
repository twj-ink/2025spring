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

# def get_next(s):
#     n=len(s)
#     nxt=[0]*n
#     j=-1
#     for i in range(1,n):
#         while j!=-1 and s[i]!=s[j+1]:
#             j=nxt[j]
#         if s[i]==s[j+1]:
#             j+=1
#         nxt[i]=j
#     return nxt
# s='aacecaaa'
# nxt=get_next(s)
s='aacecaaa'
n=len(s)
dp=[[True]*n for _ in range(n)]
for j in range(1,n):
    for i in range(j-1,-1,-1):
        dp[i][j]=dp[i+1][j-1] and s[i]==s[j]
# for i in dp:print(*i)
for j in range(n-1,-1,-1):
    if dp[0][j]:
        add=s[j+1:]
        s=add+s
        print(s)
        break