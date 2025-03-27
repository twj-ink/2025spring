#固定长度的窗口，先预处理第一组元素，然后向后一次遍历即可
class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: list[int], endTime: list[int]) -> int:
        n = len(startTime)
        diffs = [startTime[0]]  # 活动开始到第一个会议的空闲时间
        for i in range(1, n):
            diffs.append(startTime[i] - endTime[i - 1])  # 相邻会议之间的空闲时间
        diffs.append(eventTime - endTime[n - 1])  # 最后一个会议结束到活动结束的空闲时间

        cur = 0
        res = 0
        for i in range(k + 1):
            cur += diffs[i]
        res = cur
        for i in range(k + 1, n + 1):
            cur = cur + diffs[i] - diffs[i - k - 1]
            res = max(res, cur)

        return res