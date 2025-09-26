import sys
from collections import defaultdict, deque, Counter
from math import ceil, floor, log
from itertools import permutations
from bisect import bisect_left, bisect_right
from heapq import heapify, heappop, heappush
from collections import defaultdict
from typing import List
from copy import deepcopy
def solve():
    a = list(map(int,input().split()))
    b = list(map(int,input().split()))
    da, db = defaultdict(int), defaultdict(int)
    for i in range(0, len(a), 2):
        if a[i + 1] >= 0:
            da[a[i + 1]] += a[i]
    for i in range(0, len(b), 2):
        if b[i + 1] >= 0:
            db[b[i + 1]] += b[i]
    res = defaultdict(int)
    for d in (da, db):
        for k,v in d.items():
            res[k] += v
    ans = [[' ', val, ' ', key, ' '] for key, val in res.items() if (val != 0)]
    ans.sort(key=lambda x:-x[3])
    for i in range(len(ans)):
        ans[i] = '[' + ''.join(map(str, ans[i])) + ']'
    print(' '.join(map(str, ans)))

for _ in range(int(input())):
    solve()




