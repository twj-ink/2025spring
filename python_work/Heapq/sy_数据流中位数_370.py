import heapq
# maxq最大堆存贮较小值，minq最小堆存储较大值
maxq,minq=[],[]
n=int(input())
for _ in range(n):
    l=input()
    if l.startswith('Pu'):
        x=int(l.split()[1])
        # 优先存入maxq
        if not maxq or x<=-maxq[0]:
            heapq.heappush(maxq,-x)
        else:
            heapq.heappush(minq,x)
        # 保证平衡
        if len(maxq)-len(minq)>1:
            heapq.heappush(minq,-heapq.heappop(maxq))
        elif len(minq)-len(maxq)>1:
            heapq.heappush(maxq,-heapq.heappop(minq))
    else:
        if len(maxq)==len(minq):
            print(f'{(-maxq[0]+minq[0])/2:.1f}')
        else:
            print(f'{-maxq[0]:.1f}' if len(maxq)>len(minq) else f'{minq[0]:.1f}')