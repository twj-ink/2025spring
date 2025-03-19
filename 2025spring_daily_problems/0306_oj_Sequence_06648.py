### F ###
from heapq import heappop,heappush
def merge(a,b,n):
    heap,result,visited=[],[], {(0,0)}
    heappush(heap,(a[0]+b[0],0,0))
    while len(result)<n:
        res,i,j=heappop(heap)
        result.append(res)
        if i<n-1 and (i+1,j) not in visited:
            heappush(heap,(a[i+1]+b[j],i+1,j))
            visited.add((i+1,j))
        if j<n-1 and (i,j+1) not in visited:
            heappush(heap,(a[i]+b[j+1],i,j+1))
            visited.add((i,j+1))
    return result[:n]


t=int(input())
for _ in range(t):
    m,n=map(int,input().split())
    curr=sorted(list(map(int,input().split())))
    for _ in range(m-1):
        other=sorted(list(map(int,input().split())))
        curr=merge(curr,other,n)
    print(*curr)