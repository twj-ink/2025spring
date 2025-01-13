#滑动窗口，curr为当前窗口内的和，对于每一个右端点i，都尝试
#将左端点j右移，直到窗口长度最短且满足题目条件

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        i,j=0,0
        ans=float('inf')
        curr=0
        for j in range(len(nums)):
            curr+=nums[j]
            while i<j and curr-nums[i]>=target:
                curr-=nums[i]
                i+=1
            if curr>=target:
                ans=min(ans,j-i+1)
        return ans if ans!=float('inf') else 0