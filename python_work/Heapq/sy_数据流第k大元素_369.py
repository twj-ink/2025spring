import heapq
import sys

# 读取输入
n, k = map(int, sys.stdin.readline().split())
min_heap = []  # 维护大小为 K 的最小堆

for _ in range(n):
    op = sys.stdin.readline().strip()

    if op.startswith("Push"):
        _, x = op.split()
        x = int(x)
        heapq.heappush(min_heap, x)  # 先插入元素
        if len(min_heap) > k:  # 如果堆大小超过 K，弹出最小元素
            heapq.heappop(min_heap)

    elif op == "Print":
        print(min_heap[0] if len(min_heap) == k else -1)
