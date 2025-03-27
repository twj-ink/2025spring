# class Solution:
#     def maxProfit(self, prices: List[int]) -> int:
#         stack = [0] * len(prices)
#         stack[0] = prices[0]
#         for i in range(1, len(prices)):
#             stack[i] = min(stack[i - 1], prices[i])
#
#         ans = 0
#         for i in range(len(prices) - 1, -1, -1):
#             ans = max(ans, prices[i] - stack[i])
#
#         return ans
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        curr_min=float('inf')
        max_profit=0

        for p in prices:
            max_profit=max(max_profit, p-curr_min)
            curr_min=min(curr_min, p)

        return max_profit