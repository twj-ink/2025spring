# import queue
# #这个是多线程
# while True:
#     n,m=map(int,input().split())
#     if n==m==0:
#         break
#     q=queue.Queue(maxsize=n)
#     for i in range(1,n+1):
#         q.put(i)
#     while q.qsize()>1:
#         for _ in range(m-1):
#             q.put(q.get())
#         q.get()
#     print(q.get())
#
#使用deque是轻量级的
from collections import deque
while True:
    n,m=map(int,input().split())
    if n==m==0:
        break
    d=deque()
    for i in range(1,n+1):
        d.append(i)
    while len(d)>1:
        for _ in range(m-1):
            d.append(d.popleft())
        d.popleft()
    print(d[0])