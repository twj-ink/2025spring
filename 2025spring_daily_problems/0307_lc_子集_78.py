class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans=[]
        def dfs(nums,i,ans,curr):
            if i==len(nums):
                ans.append(curr[:])
                return
            curr.append(nums[i])
            dfs(nums,i+1,ans,curr)
            curr.pop()

            dfs(nums,i+1,ans,curr)
        dfs(nums,0,ans,[])
        return ans