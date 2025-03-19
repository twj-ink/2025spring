class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        cnt=0
        nums.sort()
        def dfs(nums,k,i,inq):
            nonlocal cnt
            if i==len(nums):
                cnt+=1
                return
            if nums[i]-k not in inq:
                inq.append(nums[i])
                dfs(nums,k,i+1,inq)
                inq.pop()

            dfs(nums,k,i+1,inq)
        dfs(nums,k,0,[])
        return cnt-1