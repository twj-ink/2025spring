class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x:x[0])
        l,r=intervals[0][0],intervals[0][1]
        result=[]
        for i in range(1,len(intervals)):
            if intervals[i][0]<=r:
                r=max(intervals[i][1],r)
            else:
                result.append([l,r])
                l,r=intervals[i][0],intervals[i][1]
        result.append([l,r])
        return result