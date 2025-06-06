# KMP算法和next数组

在使用kmp之前首先应该了解一下next数组的含义。

模板1：get_next函数

```python
from typing import List
def get_next(s: str) -> List:
    n=len(s)
    nxt=[-1]*n
    j=-1
    for i in range(1,n):
        while j!=-1 and s[i]!=s[j+1]:
            j=nxt[j]
        if s[i]==s[j+1]:
            j+=1
        nxt[i]=j
    return nxt
```

考虑next数组的含义：`next[i]`表示的是`s[0..i]`的**最长公共前后缀的前缀尾端索引**，这也是为什么每次比较
是否相等时用的是`s[i]==s[j+1]`，那么**前缀的长度**就是`next[i]+1`。

这是以'aabaabaabaab'为例的next展示：

| string | a  | a | b  | a  | a  | b | a | a  | b  | a  | a  | b  |
|:------:|:--:|:-:|:--:|:--:|:--:|:-:|:-:|:--:|:--:|:--:|:--:|:--:|
|  idx   | 0  | 1 | 2  | 3  | 4  | 5 | 6 | 7  | 8  | 9  | 10 | 11 |
|  next  | -1 | 0 | -1 | 0  | 1  | 2 | 3 | 4  | 5  | 6  | 7  | 8  |

举个例子，当idx为5的时候（第二个'b'），此时next[idx]=next[5]=2，这个2即是s[0..5] (aabaab)对应的**最长公共前后缀**的前缀尾端索引，也就是s[0..2] (aab).

接下来就可以进行kmp算法，来进行字符串匹配，也就是实现find函数了。find的原版实现的O(n^2)的，很慢，kmp可以变为O(n+m)，其中n为目标的字符串长度，m为待匹配的子串长度。

注意代码中的next数组是对target得到的。

模板2：kmp函数

```python
from typing import List
def kmp(s: str, target: str, next: List) -> bool: #返回能否在s中找到target
    # next = get_next(target)
    n,m=len(s),len(target)
    j=-1
    for i in range(n):
        while j!=-1 and s[i]!=target[j+1]:
            j=next[j]
        if s[i]==target[j+1]:
            j+=1
        if j==m-1: # 到达了target的末尾
            return True
    return False
```

可以发现，对于i只进行了一遍扫描，每次发现s[i]与target[j]不相等的时候，原始的find函数会这样做：将j指针重新移到target的开头；而kmp是这样做的：对j指针作有限次的回退操作而不是重新对target遍历。
这样就由find的O(n*m)降低到了O(n)，再加上next构建的O(m)，总体为O(n+m)。

例题：

### lc-找出字符串中第一个匹配项的下标-28

https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/

```python
class Solution:
    def strStr(self, s: str, target: str) -> int:
        def kmp(s,t,next):
            n,m=len(s),len(t)
            j=-1
            for i in range(n):
                while j!=-1 and s[i]!=t[j+1]:
                    j=next[j]
                if s[i]==t[j+1]:
                    j+=1
                if j==m-1:
                    return i-j
            return -1

        # next[i]表示s[0..i]对应的最长前后缀的前缀索引，即s[0..next[i]]为最长前缀，长度为next[i]+1
        def get_next(s):
            n=len(s)
            next=[-1]*n
            j=-1
            for i in range(1,n):
                while j!=-1 and s[i]!=s[j+1]:
                    j=next[j]
                if s[i]==s[j+1]:
                    j+=1
                next[i]=j
            return next
        
        next=get_next(target)
        return kmp(s,target,next)
```

### oj-前缀中的周期-01961

http://cs101.openjudge.cn/practice/01961/

考虑next数组的含义：`next[i]`表示的是`s[0..i]`的最长公共前后缀的前缀索引，这也是为什么每次比较
是否相等时用的是`s[i]==s[j+1]`，那么前缀的长度就是`next[i]+1`。

对于字符串s，遍历每一个前缀，取这个前缀的next对应的值，假设该前缀的有周期性的，那么有以下式子成立：

$$ 前缀中的最短重复周期长度 = 前缀长度 - (next数组的对应值+1) $$

比如`abcabcabc`，取前缀`abcabcabc`，i=11，`next[i]=5`,此时最短重复周期即为next对应的字符串后面的那一部分（即删掉前面的abcabc，剩下的abc就是最小周期）。

因此构建next数组之后，只需要判断这个最短重复周期长度是否是前缀的因数即可。

```python
def get_next(s):
    n=len(s)
    next=[-1]*n
    j=-1
    for i in range(1,n):
        while j!=-1 and s[i]!=s[j+1]:
            j=next[j]
        if s[i]==s[j+1]:
            j+=1
        next[i]=j
    return next

case=0
while True:
    n=int(input())
    if n==0:
        break
    s=input()
    case+=1
    print(f'Test case #{case}')
    next=get_next(s)
    for i in range(1,n):
        prefix_len = i + 1
        longest_pre_suf = next[i] + 1
        short_period = prefix_len - longest_pre_suf
        if short_period != prefix_len and prefix_len % short_period == 0:
            print(f'{prefix_len} {prefix_len // short_period}')
    print()

```

### lc-最长快乐前缀-1392

https://leetcode.cn/problems/longest-happy-prefix/

就是计算next数组的题目，然后对于这个字符串的最长前后缀的长度就是`next[-1]+1`，经过上面那道题的洗礼后这道题就是ez了。

```python
class Solution:
    def longestPrefix(self, s: str) -> str:
        def get_next(s):
            next=[-1]*len(s)
            n=len(s)
            j=-1
            for i in range(1,n):
                while j!=-1 and s[i]!=s[j+1]:
                    j=next[j]
                if s[i]==s[j+1]:
                    j+=1
                next[i]=j
            return next
        next=get_next(s)
        return s[:next[-1]+1]
```

### lc-最短回文串-214

https://leetcode.cn/problems/shortest-palindrome/

> 给定一个字符串 s，你可以通过在字符串前面添加字符将其转换为回文串。找到并返回可以用这种方式转换的最短回文串。
>
> 示例 1：

>> 输入：s = "aacecaaa"
>> 输出："aaacecaaa"

>示例 2：

>>输入：s = "abcd"
>>输出："dcbabcd"
>
>提示：
0 <= s.length <= 5 * 104
s 仅由小写英文字母组成

注意数据量，采用dp来预处理所有子串是否回文，然后再逆向遍历j找到第一个`s[0..j]`为回文子串，将后缀反转加到前方的思路会超时（做了空间优化还是超时）

该题的巧妙做法是：

$$ 找出s的最长回文前缀子串 \rightarrow 找出s+reversed(s) 的最长公共前后缀 $$

此时可以采用KMP算法，有效降低了时间复杂度。

```python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        ### KMP算法（实际上只用到了get_next函数）
        def get_next(s):
            n=len(s)
            next=[-1]*n
            j=-1
            for i in range(1,n):
                while j!=-1 and s[i]!=s[j+1]:
                    j=next[j]
                if s[i]==s[j+1]:
                    j+=1
                next[i]=j
            return next
        
        s_rev=s+'#'+s[::-1]
        next=get_next(s_rev)
        length=next[-1]+1
        suf=s[length:]
        return suf[::-1]+s

        ### 优化空间的dp（压缩为两个dp滚动数组）
        # n=len(s)
        # pre, cur=[True]*n, [True]*n
        # first=[True]
        # for j in range(1,n):
        #     for i in range(j-1,-1,-1):
        #         cur[i]=pre[i+1] and s[i]==s[j]
        #     first.append(cur[0])
        #     pre,cur=cur,[True]*n
        # for j in range(n-1,-1,-1):
        #     if first[j]:
        #         return s[j+1:][::-1]+s
        # return ''
        
        ### 未空间优化的dp
        # dp=[[True]*n for _ in range(n)]
        # for j in range(1,n):
        #     for i in range(j-1,-1,-1):
        #         dp[i][j]=dp[i+1][j-1] and s[i]==s[j]
        # # for i in dp:print(*i)
        # for j in range(n-1,-1,-1):
        #     if dp[0][j]:
        #         add=s[j+1:][::-1]
        #         s=add+s
        #         return s
        # return ''

```

### lc-找出数组中的美丽下标II-3008

https://leetcode.cn/problems/find-beautiful-indices-in-the-given-array-ii/

做了这道题之后考虑对kmp模板函数的优化。

第一步：找出a和b在s中出现位置的索引列表idxa，idxb；

第二步：遍历每一个idxa中的索引，查看idxb中是否有满足条件的j使得i符合答案，符合就加入ans中，使用二分进行。

在第一步中需要将 **所有的匹配索引保存** ，与先前的 **是否存在匹配子字符串** 要求更高，在kmp函数中做出如下优化：

1. 若m的长度仅为1，直接找出n中该字符串的索引并返回；

2. 当完成一次子字符串的匹配时(j==m-1)，将当前的i-j（匹配子字符串的开始索引）加入答案，j跳转到`next[j]`。

3. 发现其实get_next和kmp的j初始化都可以直接赋值为-1，因为第一次while循环中，j如果初始化为1，势必会进行`j=next[j]=next[0]=-1`，那就和直接初始化为-1是一致的了。


```python
from typing import List
from bisect import bisect_left
class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def get_next(s):
            n = len(s)
            next = [-1] * n
            j = -1
            for i in range(1, n):
                while j != -1 and s[i] != s[j + 1]:
                    j = next[j]
                if s[i] == s[j + 1]:
                    j += 1
                next[i] = j
            return next

        def kmp(s, a, next):
            idx = []
            j = -1
            n, m = len(s), len(a)
            if m == 1: return [i for i in range(len(s)) if s[i] == a]
            for i in range(n):
                while j != -1 and s[i] != a[j + 1]:
                    j = next[j]
                if s[i] == a[j + 1]:
                    j += 1
                if j == m - 1:
                    idx.append(i - j)
                    j=next[j]
            return idx

        na, nb = get_next(a), get_next(b)
        idxa, idxb = kmp(s, a, na), kmp(s, b, nb)
        ans = []
        if not idxa or not idxb:
            return []
        for i in idxa:
            bi = bisect_left(idxb, i)
            if bi < len(idxb) and idxb[bi] - i <= k or \
                    bi > 0 and i - idxb[bi - 1] <= k:
                if i != -1:
                    ans.append(i)
        return ans
```


## 参考：
1. [知乎：如何更好地理解和掌握 KMP 算法?](https://www.zhihu.com/question/21923021/answer/281346746)
