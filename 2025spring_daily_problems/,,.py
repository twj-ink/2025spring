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
# from bisect import bisect_left,bisect_right
# from typing import List
#
# for _ in range(int(input())):
#     a,b,c=map(int,input().split())
#     dif=b-a
#     c-=dif
#     dif2=c-b
#     if dif2>=0 and dif2%3==0:
#         print('YES')
#     else:
#         print('NO')
#
#
# from heapq import heappop,heappush,heapify
# from collections import defaultdict
# for _ in range(int(input())):
#     n=int(input())
#     s=list(map(int,input().split()))
#     heap=[s[0]]
#     ans=[]
#     total=sum(s)
#     ans.append(total)
#     for i in range(n-1):
#         heappush(heap,s[i+1])
#         total-=heap[0]
#         heappop(heap)
#         ans.append(total)
#     print(*ans[::-1])

# for _ in range(int(input())):
#     n=int(input())
#     s=input()
#     if s.count('B')==1 and s[-1]=='B':
#         print('Alice')
#     elif s.count('A')>1:
#         if s[-2:]=='AA' or s[0]==s[-1]=='A':
#             print('Alice')
#         else:
#             print('Bob')
#     else:
#         print('Bob')

def euler_sieve(n):
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False  # 0和1不是质数

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)  # i 是质数
        for prime in primes:
            if i * prime > n:
                break
            is_prime[i * prime] = False
            # 如果 prime 是 i 的最小质因数，停止继续筛选
            if i % prime == 0:
                break

    return primes
MAX=10**7
primes=euler_sieve(MAX+1)
l=len(primes)
pre=[0]*(l+1)
for i in range(1,l):
    pre[i]=primes[i-1]+pre[i-1]
for _ in range(int(input())):
    n=int(input())
    s=list(map(int,input().split()))
    curr=sum(s)
    ss=sorted(s,reverse=True)
    upper=pre[n]
    ans=0
    while upper>curr and n>=1 and ss:
        # print((upper,curr))
        n-=1
        ans+=1
        curr-=ss.pop()
        upper=pre[n]
    print(ans)