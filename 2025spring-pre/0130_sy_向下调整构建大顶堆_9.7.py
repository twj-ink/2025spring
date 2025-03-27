class BinHeap:
    def __init__(self):
        self.heapList=[0]
        self.size=0

    def buildHeap(self,alist):
        self.heapList=[0]+alist
        self.size=len(alist)
        i=len(alist)//2
        while i>0:
            self.adjustDown(i)
            i-=1

    def adjustDown(self,i):
        while i*2<=self.size:
            mc=self.maxChild(i)
            if self.heapList[mc]>self.heapList[i]:
                tmp=self.heapList[i]
                self.heapList[i]=self.heapList[mc]
                self.heapList[mc]=tmp
            i=mc

    def maxChild(self,i):
        if i*2+1>self.size:
            return i*2
        if self.heapList[i*2]<self.heapList[i*2+1]:
            return i*2+1
        return i*2

input()
alist=list(map(int,input().split()))
heap=BinHeap()
heap.buildHeap(alist)
print(*heap.heapList[1:])