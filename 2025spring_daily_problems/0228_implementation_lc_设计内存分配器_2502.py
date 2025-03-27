from collections import defaultdict

# 初始化一个内存，用cnt保存每个数字的个数，用empty来保存当前剩余空格
# allocate：如果当前的空位不够直接返回-1；否则遍历，遇到一个空位就判断从这一位开始的接下来size个是不是都是空位
# 如果是就都填上，如果不是就继续遍历
# freeMemory：答案就是d中的个数，然后遍历把相应位置重新填为0
class Allocator:

    def __init__(self, n: int):
        self.size=n
        self.mem=[0]*n
        self.cnt=defaultdict(int)
        self.empty=n

    def allocate(self, size: int, mID: int) -> int:
        if size>self.empty:
            return -1
        l,r=-1,0
        while l<self.size-1:
            l+=1
            if self.mem[l]==0:
                if l+size<=self.size and all(self.mem[r]==0 for r in range(l,l+size)):
                    for r in range(l,l+size):
                        self.mem[r]=mID
                    self.cnt[mID]+=size
                    self.empty-=size
                    return l
        return -1

    def freeMemory(self, mID: int) -> int:
        cnt=self.cnt[mID]
        self.cnt[mID]=0
        for i in range(self.size):
            if self.mem[i]==mID:
                self.mem[i]=0
        self.empty+=cnt
        return cnt


# Your Allocator object will be instantiated and called as such:
# obj = Allocator(n)
# param_1 = obj.allocate(size,mID)
# param_2 = obj.freeMemory(mID)


# 用empty来记录当前的空位区间
# 放数字时，遍历空位区间查看能不能放得下，用贪心的思路从左往右放
# 清除数字时，用dic中保存的数字的填充区间来得到答案，并将这些区间放到empty中
# 再对empty排序和合并
class Allocator:

    def __init__(self, n: int):
        self.empty = [[0, n]]
        self.dic = collections.defaultdict(list)

    def allocate(self, size: int, mID: int) -> int:
        for i in range(len(self.empty)):
            t = self.empty[i][1] - self.empty[i][0]
            if t >= size:
                ans = self.empty[i][0]
                self.dic[mID].append([self.empty[i][0], self.empty[i][0] + size])
                self.empty[i][0] = self.empty[i][0] + size
                return ans
        return -1

    def freeMemory(self, mID: int) -> int:
        count = 0
        for i in self.dic[mID]:
            count += i[1] - i[0]
            self.empty.append(i)
        self.dic[mID] = [].copy()
        self.empty.sort()
        i = 1
        while i < len(self.empty):
            if self.empty[i][0] == self.empty[i - 1][1]:
                self.empty[i - 1][1] = self.empty[i][1]
                self.empty.pop(i)
            else:
                i += 1
        return count

# Your Allocator object will be instantiated and called as such:
# obj = Allocator(n)
# param_1 = obj.allocate(size,mID)
# param_2 = obj.freeMemory(mID)