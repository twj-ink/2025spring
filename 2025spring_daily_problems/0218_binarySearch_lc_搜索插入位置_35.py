class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # return bisect_left(nums,target)
        l,r=0,len(nums)
        while l<=r:
            mid=(l+r)//2
            if l==r:
                return l
            if nums[mid]<target:
                l=mid+1
            elif nums[mid]>target:
                r=mid
            else:
                return mid