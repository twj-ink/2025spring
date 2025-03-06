class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        ans=[]
        for i in range(len(grid)):
            curr=grid[i]
            curr.sort(reverse=True)
            ans.extend(curr[:limits[i]])
        ans.sort(reverse=True)
        return sum(ans[:k])