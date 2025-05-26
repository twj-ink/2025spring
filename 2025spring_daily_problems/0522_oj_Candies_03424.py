from collections import defaultdict,deque
from heapq import heappush,heappop

n, m = map(int, input().split())
g = defaultdict(list)
for _ in range(m):
    u, v, w = map(int, input().split())
    g[u].append((v,w))

heap = []
heappush(heap, (0,1))
dist = [float('inf')] * (n + 1)
dist[1] = 0
while True:
    d, u = heappop(heap)
    if u == n:
        print(d)
        break
    if d > dist[u]:
        continue

    for v, w in g[u]:
        if dist[u] + w < dist[v]:
            dist[v] = dist[u] + w
            heappush(heap, (dist[v], v))
