class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        # return list(permutations(nums))
        def dfs(nums, visited, ans, curr):
            if len(curr) == len(nums):
                ans.append(curr[:])
                return

            for k in range(len(nums)):
                if not visited[k]:
                    curr.append(nums[k])
                    visited[k] = True
                    dfs(nums, visited, ans, curr)
                    curr.pop()
                    visited[k] = False

        ans = []
        dfs(nums, [False] * (len(nums)), ans, [])
        return ans