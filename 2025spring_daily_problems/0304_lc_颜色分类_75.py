# 双指针
def sortColors(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    left, right, i = 0, len(nums) - 1, 0
    while i <= right:
        if nums[i] == 0:
            nums[i],nums[left] = nums[left],nums[i]
            i+=1
            left+=1
        elif nums[i] == 2:
            nums[i],nums[right] = nums[right],nums[i]
            right-=1
        else:
            i+=1

# 三指针
def sortColors(self, nums: List[int]) -> None:
        last_0, last_1, last_2 = -1, -1, -1

        for num in nums:
            if num == 0:
                last_0 += 1
                last_1 += 1
                last_2 += 1
                nums[last_2] = 2
                nums[last_1] = 1
                nums[last_0] = 0
            elif num == 1:
                last_1 += 1
                last_2 += 1
                nums[last_2] = 2
                nums[last_1] = 1
            else:
                last_2 += 1
                nums[last_2] = 2