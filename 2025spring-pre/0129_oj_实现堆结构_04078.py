# 使用列表来构建堆，满足：父节点要小于两个子节点
# 当前位置的i，其父节点为i//2
# 初始化需要一个方法调用产生列表，并保留一个size大小的方法，不需要传入参数
# 定义函数：插入一个元素-从尾部插入并向上调整位置；
#         将元素位置向上调整
#         弹出一个元素-从头部弹出并将尾部元素移到头部，再向下调整位置
#         将元素位置向下调整-获取当前的最小孩子元素
#         获取当前的最小孩子元素
#         根据一个列表构建一个堆

class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.size = 0

    def insert(self,num):
        self.size+=1
        self.heapList.append(num)
        self.adjustUp(self.size)

    def adjustUp(self,i):
        while i//2>0:
            if self.heapList[i]<self.heapList[i//2]:
                tmp=self.heapList[i//2]
                self.heapList[i//2]=self.heapList[i]
                self.heapList[i]=tmp
            i//=2

    def popMin(self):
        popped=self.heapList[1]
        self.heapList[1]=self.heapList[self.size]
        self.size-=1
        self.heapList.pop()
        self.adjustDown(1)
        return popped

    def adjustDown(self,i):
        while i*2<=self.size:
            mc=self.minChild(i)
            if self.heapList[mc]<self.heapList[i]:
                tmp=self.heapList[i]
                self.heapList[i]=self.heapList[mc]
                self.heapList[mc]=tmp
            i=mc

    def minChild(self,i):
        if i*2+1>self.size:
            return i*2
        else:
            if self.heapList[i*2]<self.heapList[i*2+1]:
                return i*2
            else:
                return i*2+1

    def buildHeap(self,alist):
        i=len(alist)//2 # 中间位置之后的都是叶子节点，对之前的从下到上向下调整
        self.size=len(alist)
        self.heapList=[0]+alist
        while i>0:
            self.adjustDown(i)
            i-=1

heap=BinHeap()
n=int(input())
for _ in range(n):
    l=input()
    if l=='2':
        print(heap.popMin())
    else:
        _,u=map(int,l.split())
        heap.insert(u)
