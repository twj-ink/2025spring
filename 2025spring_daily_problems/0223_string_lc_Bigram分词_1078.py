# class Solution:
#     def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
#         text = list(map(str, text.split()))
#         n = len(text)
#
#         ans = []
#         for i in range(n - 2):
#             if text[i] == first and text[i + 1] == second:
#                 ans.append(text[i + 2])
#         return ans
from typing import List
class Solution:
    def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
        ans=[]
        for i in range(len(grid)):
            curr=grid[i]
            curr.sort(reverse=True)
            ans.extend(curr[:limits[i]])
        ans.sort(reverse=True)
        return sum(ans[:k])

s=Solution()
print(s.maxSum([[1,2],[3,4]],[1,2],2))