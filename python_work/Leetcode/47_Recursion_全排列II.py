class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums,visited,ans,curr):
            if len(curr)==len(nums):
                ans.append(curr[:])
                return
            for k in range(len(nums)):
                if k>=1 and nums[k]==nums[k-1] and visited[k-1]==False:
                    continue
                if not visited[k]:
                    visited[k]=True
                    curr.append(nums[k])
                    dfs(nums,visited,ans,curr)
                    visited[k]=False
                    curr.pop()


        nums.sort()
        ans=[]
        dfs(nums,[False]*len(nums),ans,[])
        return ans