# Kadane算法，通过一次遍历数组，维护一个局部curr和全局max变量
# 来获取最后的答案。
# 当题目涉及到“对连续子数组进行操作”时，可以考虑Kadane算法，甚至前缀和
# 此题与 辅助栈 和 前缀和 的思路类似，其中后者也可以使用Kadane算法来优化
class Solution:
    def maxFrequency(self,nums,k):
        k_freq=nums.count(k) #这是基数，下面统计额外的个数
        non_k_nums=set(i for i in nums if i!=k)
        max_gain=0 #全局变量

        for non_k_num in non_k_nums: #对每一个非k数字遍历，在数组中将其变为k
            cur_max=0
            for num in nums:
                if num==non_k_num:
                    cur_max+=1
                if num==k:
                    cur_max-=1 #因为遇到k了变化后数目会减1
                if cur_max<0:
                    cur_max=0 #如果已经变为负数就舍弃前面的一段数列
                max_gain=max(max_gain,cur_max)
        return max_gain+k_freq

# 使用Kadane的前缀和
def maxSubarr(s):
    cur_max=s[0]
    max_sum=s[0]
    for i in range(len(s)):
        cur_max=max(s[i],cur_max+s[i])
        max_sum=max(max_sum,cur_max)
    return max_sum