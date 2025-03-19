# 2025spring --> Good Questions

| 缩略 |             网站             |
|:--:|:--------------------------:|
|lc|    https://leetcode.cn     |
|oj| http://cs101.openjudge.cn/ |
|cf|   https://codeforces.com   |


### CONTENT
- [1. deque双端队列](#1-deque双端队列)
- [2. 多指针](#2-多指针)
- [3. stack栈](#3-stack栈)
- [4. linked_list链表](#4-linked_list链表)
- [5. dp](#5-dp)
- [6. binary_search二分查找](#6-binary_search二分查找)

### 1. deque双端队列

#### oj-极差不小于K-29340
>给定一个整数数组 nums 和一个整数 k，请你找出 nums 中满足如下条件的最短非空子数组：
子数组中元素的 最大值 与 最小值 之差 大于或等于 k。
返回该子数组的长度；如果不存在满足条件的子数组，则返回 -1。
注意：子数组指的是数组中连续的一段元素。
输入
第一行输入一个整数 n，表示数组的长度。
第二行输入 n 个整数，表示数组 nums 的各个元素。
第三行输入一个整数 k。
输出
输出一个整数，表示满足条件的最短子数组的长度；若不存在这样的子数组，则输出 -1。
>>样例输入
5
1 3 2 4 1
3
> 
>>样例输出
2
> 
>提示
1 <= n <= 10^4

```python
from collections import deque

n = int(input())
s = list(map(int, input().split()))
k = int(input())

L = float('inf')
maxQ, minQ = deque(), deque()
l = 0

for r in range(n):
    # 维护最大值队列，保证 maxQ[0] 是窗口内最大值
    while maxQ and s[maxQ[-1]] <= s[r]:
        maxQ.pop()
    maxQ.append(r)

    # 维护最小值队列，保证 minQ[0] 是窗口内最小值
    while minQ and s[minQ[-1]] >= s[r]:
        minQ.pop()
    minQ.append(r)

    # 窗口不满足条件时，左指针右移
    while r - l + 1 >=1 and s[maxQ[0]] - s[minQ[0]] >= k:
        L = min(L, r - l + 1)
        if maxQ[0] == l:
            maxQ.popleft()
        if minQ[0] == l:
            minQ.popleft()
        l += 1  # 缩小窗口

print(L if L != float('inf') else -1)
```

```cpp
#include <iostream>
#include <vector>
#include <deque>
#include <climits>
using namespace std;

int main() {
    int n, k;
    cin >> n;
    vector<int> s(n);
    for (int i = 0; i < n; ++i) {
        cin >> s[i];
    }
    cin >> k;

    int L = INT_MAX;
    deque<int> maxQ, minQ; // 维护最大值和最小值的双端队列
    int l = 0; // 左指针

    for (int r = 0; r < n; r++) {
        // 维护最大值队列，保证 maxQ.front() 是窗口内最大值的索引
        while (!maxQ.empty() && s[maxQ.back()] <= s[r]) {
            maxQ.pop_back();
        }
        maxQ.push_back(r);

        // 维护最小值队列，保证 minQ.front() 是窗口内最小值的索引
        while (!minQ.empty() && s[minQ.back()] >= s[r]) {
            minQ.pop_back();
        }
        minQ.push_back(r);

        // 当窗口内的 max - min >= k 时，更新答案，并尝试缩小窗口
        while (r - l + 1 >= 1 && s[maxQ.front()] - s[minQ.front()] >= k) {
            L = min(L, r - l + 1);
            if (maxQ.front() == l) maxQ.pop_front(); // 移除左端的元素
            if (minQ.front() == l) minQ.pop_front();
            l++; // 左指针右移
        }
    }

    cout << (L == INT_MAX ? -1 : L) << endl;
    return 0;
}
```
#### lc-滑动窗口最大值-239
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        maxq=deque()
        for i in range(k):
            while maxq and nums[maxq[-1]]<=nums[i]:
                maxq.pop()
            maxq.append(i)
        
        ans=[]
        ans.append(nums[maxq[0]])

        l=-1
        for r in range(k,len(nums)):
            l+=1
            while maxq and nums[maxq[-1]]<=nums[r]:
                maxq.pop()
            maxq.append(r)
            if maxq[0]==l:
                maxq.popleft()
            ans.append(nums[maxq[0]])
        return ans
```

### 2. 多指针
#### lc-颜色分类-75
https://leetcode.cn/problems/sort-colors/
> 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
必须在不使用库内置的 sort 函数的情况下解决这个问题。
>
> >示例 1：
输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
>
> >示例 2：
输入：nums = [2,0,1]
输出：[0,1,2]
> 
>提示：
n == nums.length
1 <= n <= 300
nums[i] 为 0、1 或 2

```python
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
```

### 3. stack栈
#### lc-字符串编码-394
>示例 1：
输入：s = "3[a]2[bc]"
输出："aaabcbc"
>
>示例 2：
输入：s = "3[a2[c]]"
输出："accaccacc"
>
>示例 3：
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
>
>示例 4：
输入：s = "abc3[cd]xyz"
输出："abccdcdcdxyz"
```python
class Solution:
    def decodeString(self, s: str) -> str:
        # stack,res,multi=[],'',0
        # for c in s:
        #     if c=='[':
        #         stack.append((multi,res))
        #         res,multi='',0
        #     elif c==']':
        #         curr_multi,last_res=stack.pop()
        #         res=last_res+curr_multi*res
        #     elif '0'<=c<='9':
        #         multi=multi*10+int(c)
        #     else:
        #         res+=c
        # return res

        def dfs(s,i):
            res,multi='',0
            while i<len(s):
                if '0'<=s[i]<='9':
                    multi=multi*10+int(s[i])
                elif s[i]=='[':
                    i, tmp=dfs(s,i+1)
                    res+=multi*tmp
                    multi=0
                elif s[i]==']':
                    return i,res
                else:
                    res+=s[i]
                i+=1
            return res
        return dfs(s,0)
```

#### oj-今日化学论文-20140
```python
def dfs(s,i):
    res,multi='',0
    while i<len(s):
        if '0'<=s[i]<='9':
            multi=multi*10+int(s[i])
        elif s[i]=='[':
            i, tmp=dfs(s,i+1)
            res+=tmp
        elif s[i]==']':
            return i,res*multi
        else:
            res+=s[i]
        i+=1
    return res

s=input()
ans=dfs(s,0)
print(ans)

```

### 4. linked_list链表
#### lc-排序链表-148
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def getmid(head):
            slow, fast = head, head
            while fast.next and fast.next.next:
                fast = fast.next.next
                slow = slow.next
            return slow
        
        def mergeSort(head):
            if not head or not head.next:
                return head
            mid = getmid(head)
            right_head = mid.next
            mid.next = None

            left = mergeSort(head)
            right = mergeSort(right_head)

            return merge(left, right)

        def merge(l1, l2):
            dummy = ListNode(0)
            curr = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    curr.next, l1 = l1, l1.next
                else:
                    curr.next, l2 = l2, l2.next
                curr = curr.next
            
            curr.next = l1 if l1 else l2
            return dummy.next
        
        return mergeSort(head)
```

#### lc-合并K个升序链表
https://leetcode.cn/problems/merge-k-sorted-lists/
```python
from typing import List,Optional
from heapq import heappush,heappop
#归并排序
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        def merge(l1,l2):
            dummy = ListNode(0)
            curr = dummy
            while l1 and l2:
                if l1.val <= l2.val:
                    curr.next, l1 = l1, l1.next
                else:
                    curr.next, l2 = l2, l2.next
                curr = curr.next
            curr.next = l1 if l1 else l2
            return dummy.next

        def mergeSort(lists):
            n=len(lists)
            if n==1:
                return lists[0]
            if n==0:
                return None
            mid=n//2
            left=mergeSort(lists[:mid])
            right=mergeSort(lists[mid:])
            return merge(left,right)
        
        return mergeSort(lists)

#堆
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        def __lt__(self,other):
            return self.val<other.val
        ListNode.__lt__=__lt__
        heap=[]
        for i,l in enumerate(lists):
            if l:
                heappush(heap,(l.val, i, l))
        dummy=ListNode(0)
        curr=dummy
        while heap:
            val,i,l=heappop(heap)
            curr.next=ListNode(val)
            curr=curr.next
            if l.next:
                l=l.next
                heappush(heap,(l.val,i+1,l))
        return dummy.next
```

### 5. dp
#### oj-Palindrome-01159
也涉及到了回文串，重点是找到递推公式
```python
# 超内存

'''
n=int(input())
s=input()
dp=[[0]*n for _ in range(n)]
for j in range(n):
    for i in range(j,-1,-1):
        if i>=j: continue
        if s[i]==s[j]:
            dp[i][j]=dp[i+1][j-1]
        else:
            dp[i][j]=min(dp[i+1][j]+1, dp[i][j-1]+1, dp[i+1][j-1]+2)

print(dp[0][-1])
'''

# curr存的是dp[0..n][j]这一当前列，
# prev存的是dp[0..n][j-1]这一先前列，
# 每次用到的就只有当前格子的左边、下边、左下边三个位置，即
# dp[i][j-1]   ->  prev[i]
# dp[i+1][j]   ->  curr[i+1]
# dp[i+1][j-1] ->  prev[i+1]

#也就是说：取j的时候就是curr，取j-1的时候就是prev，i的值保持不变

n = int(input())
s = input()

prev = [0] * n  # 存储上一行 dp[i+1][j]
curr = [0] * n  # 存储当前行 dp[i][j]

for j in range(1, n):  # 右端点 j 从 1 到 n-1
    for i in range(j - 1, -1, -1):  # 左端点 i 逆序遍历
        if s[i] == s[j]:
            curr[i] = prev[i + 1] if i + 1 <= j - 1 else 0
        else:
            curr[i] = min(prev[i] + 1, curr[i + 1] + 1, prev[i + 1] + 2 if i + 1 <= j - 1 else float('inf'))

    prev, curr = curr, prev  # 滚动数组，交换 prev 和 curr

print(prev[0])  # 最终答案存储在 prev[0] -> dp[0][j-1] -> dp[0][n-1]
```

#### cf-GameWithTriangles:Season2-G
https://codeforces.com/contest/2074/problem/G
将环形的点变成一条直线上的点，面积不能重叠->区间dp
`dp[l][r]=max(dp[l+1][i-1]+dp[i+1][r-1]+s[l]*s[i]*s[r], dp[l][i]+dp[i+1][r])`
同样也是考虑：选两端点？不选两端点？其中选的时候i从l+1开始，不选的时候i从l开始
```python
def solve():
    n=int(input())
    s=list(map(int,input().split()))
    dp=[[0]*n for _ in range(n)]
    for r in range(2,n):
        for l in range(r-2,-1,-1):
            for i in range(l+1,r):
                dp[l][r]=max(dp[l][r],dp[l+1][i-1]+dp[i+1][r-1]+s[l]*s[i]*s[r],dp[l][i]+dp[i+1][r])
            for i in range(l,r):
                dp[l][r]=max(dp[l][r],dp[l][i]+dp[i+1][r])
    print(dp[0][-1])


def main():
    t=int(input())
    for _ in range(t):
        solve()

main()
```

```cpp
#include <iostream>
#include <vector>
#include <algorithm>
using ll = long long;
using namespace std;

void solve(){
    int n;
    cin >> n;
    vector<int> s(n);
    for (int i=0;i<n;i++){
        cin >> s[i];
    }

    vector dp(n, vector<ll>(n,0));
    for (int r = 2;r < n; r++){
        for (int l = r - 2; l >= 0; l--){
            for (int i = l + 1; i < r; i++){
                dp[l][r] = max(dp[l][r], dp[l + 1][i - 1] + dp[i + 1][r - 1] + s[l] * s[i] * s[r]);
            }
            for (int i = l; i < r; i++){
                dp[l][r] = max(dp[l][r], dp[l][i] + dp[i + 1][r]);
            }
        }
    }

    cout << dp[0][n - 1] << endl;
};

int main(){
    int t;
    cin >> t;
    while (t--){
        solve();
    }
    return 0;
}
```

### 6. binary_search二分查找
#### cf-TwoColors-2075C
https://codeforces.com/contest/2075/problem/C
使用`m-bisect_left(a,k)`找到能涂k个板子的颜色种类数目，然后初步数目为x*y，假设k>n-k，那么能涂k块的x种颜色一定也可以涂(n-k)块，也就是说x中包含了y，所以要减去min(x,y)，这些是用同一种颜色涂的方案数。
```python
from bisect import bisect_left

for _ in range(int(input())):
    n, m = map(int, input().split())
    a = sorted(list(map(int, input().split())))
    ans = 0
    for k in range(1, n):
        x = m - bisect_left(a, k)
        y = m - bisect_left(a, n - k)
        ans += x * y - min(x, y)
    print(ans)
```