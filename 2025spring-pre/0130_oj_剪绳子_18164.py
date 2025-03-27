from heapq import heappop,heappush,heapify
n=int(input())
s=list(map(int,input().split()))
heapify(s)
ans=0
while len(s)>1:
    a=heappop(s)
    b=heappop(s)
    ans+=(a+b)
    heappush(s,a+b)
print(ans)