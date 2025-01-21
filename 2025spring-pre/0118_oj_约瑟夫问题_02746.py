import queue

while True:
    n,m=map(int,input().split())
    if n==m==0:
        break
    q=queue.Queue(maxsize=n)
    for i in range(1,n+1):
        q.put(i)
    while q.qsize()>1:
        for _ in range(m-1):
            q.put(q.get())
        q.get()
    print(q.get())