class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        items.sort(key=lambda x: x[0])
        idx = sorted(range(len(queries)), key=lambda x: queries[x])

        ans = [0] * len(queries)
        max_beauty, j = 0, 0
        for i in idx:
            q = queries[i]
            while j < len(items) and items[j][0] <= q:
                max_beauty = max(max_beauty, items[j][1])
                j += 1
            ans[i] = max_beauty
        return ans

        # n=len(queries)

        # d=defaultdict(list)
        # for p,b in items:
        #     d[p].append(b)
        # last_idx=0
        # last_val=0

        # p=sorted([i for (i,_) in items])
        # ans=[0]*n
        # q=sorted([(v,i) for i,v in enumerate(queries)])
        # for v,i in q:
        #     curr=bisect_right(p,v)
        #     if curr == last_idx:
        #         ans[i]=last_val
        #         continue
        #     for j in range(last_idx,curr):
        #         for b in d[p[j]]:
        #             last_val=max(last_val,b)
        #             ans[i]=max(ans[i],last_val)
        #     last_idx=curr
        #     # last_val=ans[i]

        # return ans