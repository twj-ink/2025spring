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
from typing import List
def get_next(s: str) -> List:
    n=len(s)
    nxt=[-1]*n
    j=-1
    for i in range(1,n):
        while j!=-1 and s[i]!=s[j+1]:
            j=nxt[j]
        if s[i]==s[j+1]:
            j+=1
        nxt[i]=j
    return nxt
def kmp(s: str, target: str, next: List) -> bool: #返回能否在s中找到target
    # next = get_next(target)
    n,m=len(s),len(target)
    j=0
    for i in range(n):
        while j!=-1 and s[i]!=target[j+1]:
            j=next[j]
        if s[i]==target[j+1]:
            j+=1
        if j==m-1: # 到达了target的末尾
            return True
    return False
s='aabaabaabaab'
target='aac'
nxt=get_next(target)
print(kmp(s,target,nxt))

