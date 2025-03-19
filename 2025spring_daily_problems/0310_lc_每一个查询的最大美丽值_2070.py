class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        n = len(queries)

        d = defaultdict(list)
        for p, b in items:
            d[p].append(b)
        last_idx = 0
        last_val = 0

        p = sorted([i for (i, _) in items])
        ans = [0] * n
        q = sorted([(v, i) for i, v in enumerate(queries)])
        for v, i in q:
            curr = bisect_right(p, v)
            if curr == last_idx:
                ans[i] = last_val
                continue
            for j in range(last_idx, curr):
                for b in d[p[j]]:
                    last_val = max(last_val, b)
                    ans[i] = max(ans[i], last_val)
            last_idx = curr
            # last_val=ans[i]

        return ans