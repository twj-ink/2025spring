# # 分治排序
# def merge_sort(s):
#     if len(s)<=1:
#         return s
#     mid=len(s)//2
#     left=merge_sort(s[:mid])
#     right=merge_sort(s[mid:])
#     return Merge(left,right)
# def Merge(left,right):
#     i=j=0
#     ans=[]
#     while i<len(left) and j<len(right):
#         if left[i]<right[j]:
#             ans.append(left[i])
#             i+=1
#         else:
#             ans.append(right[j])
#             j+=1
#     ans.extend(left[i:])
#     ans.extend(right[j:])
#     return ans
#
# s=[4,2,6,3,8,1]
# print(merge_sort(s))
#
# # 快速排序--递归写法
# def quick_sort(s):
#     if len(s)<=1:
#         return s
#     pi=s[0]
#     left=[i for i in s[1:] if i<pi]
#     right=[i for i in s[1:] if i>=pi]
#     return quick_sort(left)+[pi]+quick_sort(right)
#
# s=[4,2,6,3,8,1]
# print(quick_sort(s))
#
# # 快速排序--双指针写法
# def quickSort(s,l,r):
#     if l<r:
#         #partition_pos是选取的基准在排好序之后的位置，方便后续进行递归
#         partition_pos=partition(s,l,r)
#         quickSort(s,l,partition_pos-1)
#         quickSort(s,partition_pos+1,r)
#
# def partition(s,l,r):
#     i,j=l,r-1
#     pivot=s[r] # 选择最右侧为基准
#     while i<=j:
#         while i<=r and s[i]<pivot:
#             i+=1  # 从左往右找到第一个大于基准的值
#         while j>=l and s[j]>pivot:
#             j-=1  # 从右往左找到第一个小于基准的值
#         if i<j:
#             s[i],s[j]=s[j],s[i] # 如果两者位置不符合条件就互换
#     if s[i]>pivot:
#         s[i],s[r]=s[r],s[i] # 由于s[i]可能是第一个大于基准的值，所以要把它与基准互换
#     return i # 基准现在的位置
#
# s=[4,2,6,3,8,1]
# quickSort(s,0,len(s)-1)
# print(s)
#
# # 分治排序
# #pylint:skip-file
# def merge_sort(s):
#     if len(s)<=1:
#         return s
#     mid=len(s)//2
#     left=merge_sort(s[:mid])
#     right=merge_sort(s[mid:])
#     return Merge(left,right)
# def Merge(left,right):
#     global cnt
#     i=j=0
#     ans=[]
#     while i<len(left) and j<len(right):
#         if left[i]<right[j]:
#             ans.append(left[i])
#             i+=1
#         else:
#             ans.append(right[j])
#             j+=1
#             cnt+=len(left)-i
#     ans.extend(left[i:])
#     ans.extend(right[j:])
#     return ans
#
# while True:
#     n=int(input())
#     if n==0:
#         break
#     s=[]
#     cnt=0
#     for _ in range(n):
#         s.append(int(input()))
#     merge_sort(s)
#     print(cnt)

# 快速排序--双指针写法
#pylint:skip-file
def quickSort(s,l,r):
    if l<r:
        #partition_pos是选取的基准在排好序之后的位置，方便后续进行递归
        partition_pos=partition(s,l,r)
        quickSort(s,l,partition_pos-1)
        quickSort(s,partition_pos+1,r)

def partition(s,l,r):
    global cnt
    i,j=l,r-1
    pivot=s[r] # 选择最右侧为基准
    while i<=j:
        while i<=r and s[i]<pivot:
            i+=1  # 从左往右找到第一个大于基准的值
        while j>=l and s[j]>pivot:
            j-=1  # 从右往左找到第一个小于基准的值
        if i<j:
            s[i],s[j]=s[j],s[i] # 如果两者位置不符合条件就互换
            cnt+=1
    if s[i]>pivot:
        s[i],s[r]=s[r],s[i] # 由于s[i]可能是第一个大于基准的值，所以要把它与基准互换
        cnt+=1
    return i # 基准现在的位置

while True:
    n=int(input())
    if n==0:
        break
    s=[]
    cnt=0
    for _ in range(n):
        s.append(int(input()))
    quickSort(s,0,len(s)-1)
    print(cnt)