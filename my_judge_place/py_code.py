import sys
from collections import defaultdict, deque, Counter
from math import ceil, floor, log
from itertools import permutations
from bisect import bisect_left, bisect_right
from heapq import heapify, heappop, heappush
from collections import defaultdict
from typing import List
from copy import deepcopy

def sliding(arr, k):
    n = len(arr)
    # minq维护严格递增的下标队列，保证队头始终是最小值的下标
    minq, maxq = deque(), deque()
    mins, mans = [], []
    if n == 0 or k == 0:
        return [], []

    for i in range(n):
        # 保证minq是严格递增的
        while minq and arr[minq[-1]] >= arr[i]:
            minq.pop()
        minq.append(i)

        # 保证maxq是严格递减的
        while maxq and arr[maxq[-1]] <= arr[i]:
            maxq.pop()
        maxq.append(i)

        # 移除左侧元素，根据窗口左侧的索引来确定很方便
        if minq[0] <= i - k:
            minq.popleft()
        if maxq[0] <= i - k:
            maxq.popleft()
        
        # 当 i >= k-1 时窗口形成
        if i >= k - 1:
            mins.append(arr[minq[0]])
            maxs.append(arr[maxq[0]])
    
    return mins, maxs

n, k = map(int,input().split())
arr = list(map(int, input().split()))
mins, maxs = sliding(arr, k)
print(*mins)
print(*maxs)
