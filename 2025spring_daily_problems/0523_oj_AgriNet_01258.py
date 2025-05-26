from heapq import heappush,heappop

while True:
    try:
        n = int(input())
        g = []
        for _ in range(n):
            g.append(list(map(int, input().split())))
        visited = [False] * n
        heap = []
        heappush(heap, (0, 0)) # (weight, node)
        ans = 0
        while heap:
            d, u = heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            ans += d
            for v in range(n):
                if not visited[v]:
                    heappush(heap, (g[u][v], v))
        print(ans)
    except EOFError:
        break