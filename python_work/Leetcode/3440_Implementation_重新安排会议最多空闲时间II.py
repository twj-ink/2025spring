class Solution:
    def maxFreeTime(self,eventTime,startTime,endTime):
        n=len(startTime)

        #计算每个会议左边的最大空缺时间
        lRoom=[0]*n
        lRoom[0]=startTime[0]
        for i in range(1,n):
            lRoom[i]=max(lRoom[i-1],startTime[i]-endTime[i-1])

        #计算每个会议右边的最大空缺时间
        rRoom=[0]*n
        rRoom[-1]=eventTime-endTime[-1]
        for i in range(n-2,-1,-1):
            rRoom[i]=max(rRoom[i+1],startTime[i+1]-endTime[i])

        #下面开始计算每个会议左右的间隔之和以及是否可以移动这个会议
        res=0
        for i in range(n):
            lTime=0 if i==0 else endTime[i-1]
            rTime=eventTime if i==n-1 else startTime[i+1]

            cur_length=endTime[i]-startTime[i]

            res=max(res,rTime-lTime-cur_length)
            
            #如果左侧或者右侧有可以放这个会议的间隔
            #只考虑能不能，所以lRoom需要保持最大值
            if i>0 and lRoom[i-1]>=cur_length:
                res=max(res,rTime-lTime)
            if i<n-1 and rRoom[i+1]>=cur_length:
                res=max(res,rTime-lTime)
        return res