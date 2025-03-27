from heapq import heappop,heappush,heapify
from typing import List
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        cnt=0
        if len(nums) in [0,1]:
            return 0
        heapify(nums)
        while len(nums)>1 and nums[0]<k:
            a=heappop(nums)
            b=heappop(nums)
            heappush(nums,min(a,b)*2+max(a,b))
            cnt+=1
        return cnt