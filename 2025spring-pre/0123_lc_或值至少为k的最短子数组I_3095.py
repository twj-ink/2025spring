###滑动窗口
#由于按位或没有逆运算，在左窗口向右移动时的操作无法处理
#可以考虑使用一个bit数组统计当前窗口内二进制的1的个数，用s来记录当前窗口内的或值

#当右侧进入时，bit中会存在0->1的变化，其余的1会累积，
#而窗口内的或值s仅仅会因0->1的变化而增大(或运算是递增的)，因此用辗转相除来用位运算对s操作
#当左侧出去时，使用同样的方法对s进行位运算的操作

#下面代码的思路框架是--滑动窗口，特殊条件是--按位或，应对思路是--位运算更新或值
from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        ans=float('inf')
        left=0
        bit=[0]*32 #记录各个数字的二进制中1的累积值
        s=0

        for right,num in enumerate(nums):
            ###滑动窗口右侧加入这个元素###
            #由于按位或的结果肯定会递增,只需要对s加上bit中由0变1的数值
            #即按位或只会让bit中的1变多
            i=0
            while num>0:
                #如果第i位原来是0,且需要变为1
                if bit[i]==0 and num%2==1:
                    s+=1<<i #即s+=2**i
                bit[i]+=num%2
                num//=2
                i+=1
            ###左窗口向右移动###
            while s>=k and left<=right:
                ans=min(ans,right-left+1)
                #移除左侧元素
                num=nums[left]
                i=0
                while num>0:
                    #如果第i为只有一个1(也就是这个数字本身造成的1),就变为0
                    if bit[i]==1 and num%2==1:
                        s-=1<<i
                    bit[i]-=num%2
                    num//=2
                    i+=1
                left+=1
        return ans if ans!=float('inf') else -1

#下面代码的思路框架是--滑动窗口，特殊条件是--按位或，应对思路是--用集合知识优化n^2暴力
#对于集合可转为二进制数字，如{1,2,4} -> 1011，这样当a|b=a时说明b是a的子集，此后再求或是没有新的0->1的变化的，即或值不会再改变，可以跳出向后遍历了
class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        ans = float('inf')
        for i, x in enumerate(nums):
            if x >= k:
                return 1
            j = i - 1 #对每个遍历到的数字，从这个数字往前遍历，依次把或运算的值存在前面的各个位置
            while j >= 0 and nums[j] | x != nums[j]: #这个优化考虑的是集合的子集
                nums[j] |= x
                if nums[j] >= k:
                    ans = min(ans, i - j + 1)
                j -= 1
        return ans if ans < float('inf') else -1
#由于或运算是累计的结果，即使对nums数组原地累积求或，后续再次求是没有影响的(因为只有0->1的变化)