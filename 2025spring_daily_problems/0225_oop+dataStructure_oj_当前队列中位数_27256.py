from heapq import heappop,heappush
from collections import defaultdict
maxq,minq,maxsize,minsize=[],[],0,0
s=[]
idx=0
d=defaultdict(int)
def balance():
    global maxsize,minsize
    while maxsize-minsize > 1:
        prune(maxq)
        heappush(minq, -heappop(maxq))
        maxsize-=1; minsize+=1
    while minsize-maxsize > 1:
        prune(minq)
        heappush(maxq, -heappop(minq))
        maxsize+=1; minsize-=1

def prune(heap):
    global maxsize,minsize
    if heap is maxq:
        while heap and d[-heap[0]]:
            d[-heap[0]]-=1
            heappop(heap)
    else:
        while heap and d[heap[0]]:
            d[heap[0]]-=1
            heappop(heap)


for _ in range(int(input())):
    l=input()
    if l.startswith('add'):
        x=int(l.split()[1])
        s.append(x)
        if not maxq or x<=-maxq[0]:
            heappush(maxq,-x)
            maxsize+=1
        else:
            heappush(minq,x)
            minsize+=1
        balance()

    elif l=='del':
        d[s[idx]]+=1
        if maxq and s[idx]<=-maxq[0]:
            maxsize-=1
            if s[idx]==-maxq[0]:
                prune(maxq)
        else:
            minsize-=1
            if minq and s[idx]==minq[0]:
                prune(minq)
        balance()
        idx+=1

    else:
        prune(maxq)
        prune(minq)
        balance()
        if maxsize==minsize:
            ans=(-maxq[0]+minq[0])//2
            if ans==(-maxq[0]+minq[0])/2:
                print(ans)
            else:
                print(f'{(-maxq[0]+minq[0])/2:.1f}')
        else:
            print(-maxq[0] if maxsize>minsize else minq[0])


