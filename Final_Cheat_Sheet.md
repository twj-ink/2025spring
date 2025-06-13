并查集
```python
parent = [i for i in range(n)]
rank = [0]*n
def find(x):
    if x != parent[x]:
        parent[x] = find(x, parent[x])
    return parent[x]
def union(x,y):
    xp,yp=find(x),find(y)
    if xp==yp:
        pass
    if rank[xp] < rank[yp]:
        parent[xp] = yp
    elif rank[xp] > rank[yp]:
        parent[yp] = xp
    else:
        parent[yp] = xp
        rank[xp] += 1
```
Trie
```python
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.cnt = 0
        self.is_end = False
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = TrieNode()
            cur = cur.children[ch]
            cur.cnt += 1
        cur.is_end = True
    def get_each_unique_prefix(self, word):
        cur = self.root
        pre = ''
        for ch in word:
            pre += ch
            cur = cur.children[ch]
            if cur.cnt == 1:
                return pre
        return pre
    def find_all_common_pre(self):
        cur = self.root
        pre = ''
        if not cur:
            return pre
        while not cur.is_end and len(cur.children) == 1:
            ch = next(iter(cur.children))
            pre += ch
            cur = cur.children[ch]
        return pre
```
KMP

next[i]表示的是s[0..i]的最长公共前后缀的前缀尾端索引，这也是为什么每次比较
是否相等时用的是s[i]==s[j+1]，那么前缀的长度就是next[i]+1。
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
palindromic
```python
def process(s):
    n=len(s)
    dp=[[True]*n for _ in range(n)]
    for j in range(n):
        for i in range(j-1,-1,-1):
            dp[i][j]=(dp[i+1][j-1] and s[i]==s[j])
def ManacherPalindrome(s: str) -> str: #马拉车最长回文子串
    t='#'+'#'.join(s)+'#'
    n=len(t)
    P=[0]*n
    C,R=0,0
    for i in range(n):
        # 找对称位置并判断是否可以初始化为对称位置的半径
        mirr=2*C-i
        if i<R:
            P[i]=min(P[mirr],R-i)
        # 拓展该位置的半径
        while i+P[i]+1<n and i-P[i]-1>=0 and t[i+P[i]+1]==t[i-P[i]-1]:
            P[i]+=1
        # 拓展后的回文串如果超出了右边界，更新
        if i+P[i]>R:
            C,R=i,i+P[i]
    max_len,center_idx=max((n,i) for i, n in enumerate(P))
    begin=(center_idx-max_len)//2
    return s[begin:begin+max_len]
```
二次探查法散列表
```python
n,m=map(int,input().split())
s=list(map(int,input().split()))
hash_table = [-1] * m
idx = []
for i in s:
    base = 0
    while True:
        pos = (i % m + base ** 2) % m
        if hash_table[pos] in (-1, i):
            hash_table[pos] = i
            idx.append(pos)
            break
        pos = (i % m - base ** 2) % m
        if hash_table[pos] in (-1, i):
            hash_table[pos] = i
            idx.append(pos)
            break
        base += 1
print(*idx)
```
欧拉筛
```python
def euler_sieve(n):
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        for prime in primes:
            if i * prime > n:
                break
            is_prime[i * prime] = False
            if i % prime == 0:
                break
    return primes
def chose(n):
    prime=[]
    isprime=[True]*(n+1)
    isprime[0]=isprime[1]=False
    for i in range(2,n+1):
        if isprime[i]:
            prime.append(i)
            for k in range(i**2,n+1,i):
                isprime[k]=False
    return prime
```
二分
```python
left,right=0,1e9
while left<=right:
    mid=(left+right)//2
    if can_reach(mid):
        right=mid-1
    else:
        left=mid+1
print(left)
```
单调栈
```python
#柱状图中最大的矩形
def largestRectangleArea(self, heights: List[int]) -> int:
    stack=[0]
    n=len(heights)
    heights=[0]+heights+[0]
    ans=0

    for i in range(1,n+2):
        while stack and heights[i]<heights[stack[-1]]:
            curr_h=heights[stack.pop()]
            curr_w=i-stack[-1]-1
            ans=max(ans,curr_h*curr_w)
        stack.append(i)
    return ans
```
卡特兰数
```python
from math import comb
def catalan_comb(n):
    return comb(2*n,n) // (n+1)
```
mergesort
```python
def merge_sort(s):
    if len(s)<=1:
        return s
    mid=len(s)//2
    left=merge_sort(s[:mid])
    right=merge_sort(s[mid:])
    return merge(left,right)
def merge(l,r):
    ans=[]
    i=j=0
    while i<len(l) and j<len(r):
        if l[i]<r[j]: ans.append(l[i]); i+=1
        else: ans.append(r[j]); j+=1
    ans.extend(l[i:])
    ans.extend(r[j:])
    return ans
```
prime MST
```python
def prim(g, n):
    heap = [(0, 0)] # (weight, vertex)
    visited = [False] * n
    mst_cost = 0
    mst_node = 0
    while heap:
        w, u = heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        mst_cost += w
        mst_node += 1
        for nei, nei_w in g[u]:
            if not visited[nei]:
                heappush(heap, (nei_w, nei))
    return mst_cost if mst_node == n else -1
```
Kruskal
```python
class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def unite(self, x, y):
        px, py = self.find(x), self.find(y)
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1
def kruskal(n, edges):
    dsu = DSU(n)
    edges.sort(key = lambda x:x[2])
    mst_cost = 0
    mst_node = 1
    for u, v, w in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.unite(u, v)
            mst_node += 1
            mst_cost += w
    return mst_cost if mst_node == n else -1
```
dijkstra  如果没有d>dist[u]可以用visited集合防止重复
```python
def dij(n, g, source):
    dist = [float('inf')] * n
    dist[source] = 0
    heap = [(0, source)] # (length, node)
    while heap:
        d, u = heapq.heappop(heap)
        # 如果遇到了目标节点直接返回
        # if u == target:
        #     return d
        # 可以直接写成while True:
        # 剪枝
        if d > dist[u]: 
            continue
        for v, w in g[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist
```
Bellman-Ford

对于负权图和负权图判环的算法。并不是dijkstra的贪心，而是暴力系统枚举所有可能的松弛路径的方法。

对于n个节点，假设每次更新一条边和一个节点，那么最多更新n-1次即可将dist数组化为最短路径。在n-1次之后的第n次如果仍能松弛说明存在环，此时就没有最短路一说了。

注意每次松弛时要把dist复制一遍，因为有可能在同一轮中，v先更新，然后这个v作为u更新下一个v已经是第二轮了。

```python
def bellman(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    # n - 1次松弛
    for _ in range(n - 1):
        clone = dist[:] # clone dist数组
        for u, v, w in edges:
            if clone[u]!=float('inf') and clone[u] + w < dist[v]:
                dist[v] = clone[u] + w
    # 第n次松弛检测是否有环
    for u, v, w in edges:
        if dist[u]!=float('inf') and dist[u] + w < dist[v]:
            return -1
    
    return dist
```

但是这样会将所有的点无脑循环，可以使用双端队列deque来将确实需要松弛的节点存入，每次确实松弛了的节点再重新放入队列中。同时用inq来维护节点是否在队列中。(SPFA)但是无法处理存在环的图。

```python
from collections import deque
def spfa(n, graph, source):
    dist = [float('inf')] * n
    dist[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        clone = dist[:]
        for _ in range(len(q)):
            for v, w in graph[u]:
                if clone[u] + w < dist[v]:
                    dist[v] = clone[u] + w
                    q.append(v)
    return dist
```
Floyd-Warshall

多源最短路，可以是负权图。初始化`dist[i][j]`表示从i到j的最短路。

第一步，dist[i][i]=0;如果i与j直接有边dist[i][j]=w;其余为inf。

第二步，遍历中间节点k，左右节点 ij，更新可能的最短路。

```python
def floyd(n, edges):
    dist = [[float('inf')] * n for _ in range(edges)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
        # dist[v][u] = w 无向图
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # 检查负权环
    for i in range(n):
        if dist[i][i] < 0:
            return -1
    
    return dist
```
拓扑排序 + 关键路径长度

本题的核心在于判断给定的有向图是否为有向无环图（DAG），并在此基础上求解关键路径长度。**关键路径**是指在一个项目中，所有活动（边）的持续时间之和最长的路径，它决定了整个项目的最短完成时间。
```python
def topo_sort(n, g, indeg):
    q = deque([i for i in range(n) if indeg[i] == 0])
    topo_order = []
    length = [0] * n
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v, w in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
            length[v] = max(length[v], length[u] + w) # 由前置活动时间更新后续活动最早时间
    return topo_order if len(topo_order) == n else -1, length
def get_critical_path_length(n, g, indeg):
    topo_order, length = topo_sort(n, g, indeg)
    if topo_order == -1:
        return -1
    return max(length)
```

要点
1. 拓扑排序 -> 正向更新 ve[u]，如果 EST[u] + weight(u, v) 大于 EST[v] ，则更新 EST[v] =
EST[u] + weight(u, v) 。
2. 逆拓扑排序 -> 反向更新 vl[u]，如果 LST[u]  weight(v, u) 小于 LST[v] ，则更新 LST[v]
= LST[u] - weight(v, u) 。
3. 若某边 (u→v) 满足 ve[u] == vl[v] - w，则是关键活动
4. 最长路径 ve[终点] = 项目总工期

```python
from collections import deque, defaultdict

def critical_path(n, edges):
    """
    关键路径分析（AOE网络）
    参数:
        n: 节点总数（编号0到n-1）
        edges: 所有边 (u, v, w)，表示从u到v需要时间w
    返回:
        project_time: 项目总工期
        critical_activities: 关键活动列表，每个是(u, v)
    """
    # Step 1. 构建邻接表 + 入度
    adj = [[] for _ in range(n)]
    indegree = [0] * n
    for u, v, w in edges:
        adj[u].append((v, w))
        indegree[v] += 1

    # Step 2. 正向拓扑排序 -> 计算 ve[]
    ve = [0] * n
    q = deque()
    for i in range(n):
        if indegree[i] == 0:
            q.append(i)

    topo_order = []
    while q:
        u = q.popleft()
        topo_order.append(u)
        for v, w in adj[u]:
            if ve[u] + w > ve[v]:
                ve[v] = ve[u] + w
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    # Step 3. 逆拓扑排序 -> 计算 vl[]
    project_time = max(ve)
    vl = [project_time] * n
    for u in reversed(topo_order):
        for v, w in adj[u]:
            if vl[v] - w < vl[u]:
                vl[u] = vl[v] - w

    # Step 4. 找关键活动
    critical_activities = []
    for u in range(n):
        for v, w in adj[u]:
            e = ve[u]
            l = vl[v] - w
            if e == l:
                critical_activities.append((u, v))

    return project_time, critical_activities

# 示例：
if __name__ == "__main__":
    n = 6
    edges = [
        (0, 1, 3),
        (0, 2, 2),
        (1, 3, 2),
        (2, 3, 1),
        (3, 4, 4),
        (3, 5, 2),
        (4, 5, 3)
    ]
    project_time, critical_activities = critical_path(n, edges)
    print(f"项目总工期：{project_time}")
    print("关键活动：")
    for u, v in critical_activities:
        print(f"({u} -> {v})")
```
无向图判环
```python
def has_cycle_undirected_dfs(graph):
    visited = set()
    def dfs(node, parent):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False
    for node in graph:
        if node not in visited:
            if dfs(node, -1):
                return True
    return False
def find(x, parent):
    if x != parent[x]:
        parent[x] = find(parent[x], parent)
    return parent[x]
def union(x, y, parent):
    root_x, root_y = find(x, parent), find(y, parent)
    if root_x == root_y:
        return True
    parent[root_x] = root_y
    return False
def has_cycle_undirected_dsu(n, edges):
    parent = [i for i in range(n + 1)]
    for u, v in edges:
        if union(u, v, parent):
            return True
    return False
```
有向图判环
```python
from collections import defaultdict,deque
def has_cycle_directed_topo(graph):
    indeg = defaultdict(int)
    for u in graph:
        for v in graph[u]:
            indeg[v] += 1
    q = deque([node for node in graph if indeg[node] == 1])
    visited_cnt = 0
    while q:
        node = q.popleft()
        visited_cnt += 1
        for neighbor in graph[node]:
            indeg[neighbor] -= 1
            if indeg[neighbor] == 0:
                q.append(neighbor)
    return visited_cnt != len(graph)
def has_cycle_directed_color(graph):
    s=[0]*len(graph)
    def dfs(i, graph):
        if s[i]==1:
            return True
        if s[i]==2:
            return False
        s[i]=1
        for nei in graph[i]:
            if dfs(nei, graph):
                return True
        s[i]=2
        return False
    for i in range(len(graph)):
        if dfs(i,graph):
            return False
    return True
```
判断是否为平衡二叉树bst
```python
# 对于所有节点，其左子树和右子树的高度差不超过1
def is_balanced(root):
    def check_height(root):
        if not root:
            return 0
        left_height=check_height(root.left)
        if left_height==-1:
            return -1
        right_height=check_height(root.right)
        if right_height==-1 or abs(left_height-right_height)>1:
            return -1
        return max(left_height,right_height)+1
    return check_height(root) != -1
```
中序转后序
```python
def infix_to_postfix(expression):
    precedence = {'+':1, '-':1, '*':2, '/':2}
    stack = []
    postfix = []
    number = ''
    for char in expression:
        if char.isnumeric() or char == '.':
            number += char
        else:
            if number:
                num = float(number)
                postfix.append(int(num) if num.is_integer() else num)
                number = ''
            if char in '+-*/':
                while stack and stack[-1] in '+-*/' and precedence[char] <= precedence[stack[-1]]:
                    postfix.append(stack.pop())
                stack.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    postfix.append(stack.pop())
                stack.pop()
    if number:
        num = float(number)
        postfix.append(int(num) if num.is_integer() else num)
    while stack:
        postfix.append(stack.pop())
    return ' '.join(str(x) for x in postfix)
n = int(input())
for _ in range(n):
    expression = input()
    print(infix_to_postfix(expression))
```
树上dp
```python
import sys
sys.setrecursionlimit(10**7)
n = int(input())
s = [0] + list(map(int, input().split()))
ans = 0
def dfs(i):
    global ans
    if i > n:
        return 0, 0
    l1, l0 = dfs(i * 2)
    r1, r0 = dfs(i * 2 + 1)
    cur1 = s[i] + l0 + r0
    cur0 = max(l1, l0) + max(r1, r0)
    ans = max(ans, cur1, cur0)
    return cur1, cur0
dfs(1)
print(ans)

#树的最大路径和
import sys
sys.setrecursionlimit(1000000)
def dfs(u, val, tree, visited, maxSum):
    visited[u] = True
    max1, max2 = 0, 0
    for v in tree[u]:
        if not visited[v]:
            t = dfs(v, val, tree, visited, maxSum)
            if t > max1:
                max2 = max1
                max1 = t
            elif t > max2:
                max2 = t
    maxSum[0] = max(maxSum[0], val[u] + max1 + max2)
    return max(0, val[u] + max1)
def main():
    T = int(sys.stdin.readline())
    for _ in range(T):
        n = int(sys.stdin.readline())
        val = [0] * (n + 1)
        tree = [[] for _ in range(n + 1)]
        visited = [False] * (n + 1)

        vals = list(map(int, sys.stdin.readline().split()))
        for i in range(1, n + 1):
            val[i] = vals[i - 1]

        for _ in range(n - 1):
            s, t = map(int, sys.stdin.readline().split())
            tree[s].append(t)
            tree[t].append(s)

        maxSum = [val[1]]  # 使用列表封装，以便在递归中修改
        dfs(1, val, tree, visited, maxSum)
        print(maxSum[0])
main()
```
骑士周游
```python
dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]
def can_move(n,s,x,y):
    return 0<=x<n and 0<=y<n and s[x][y]==0
def get_degree(x,y):
    cnt=0
    for i in range(8):
        nx,ny=x+dx[i],y+dy[i]
        if can_move(n,s,nx,ny):
            cnt+=1
    return cnt
def dfs(n,r,c,step,s,sr,sc):
    if step==n**2:
        return True
    degrees=[]
    for i in range(8):
        nx,ny=r+dx[i],c+dy[i]
        if (can_move(n,s,nx,ny) or (step==n**2-1 and (nx,ny)==(sr,sc))):
            deg=get_degree(nx,ny)
            degrees.append((deg,nx,ny))
    degrees.sort()
    for _, nx, ny in degrees:
        s[nx][ny]=1
        if dfs(n,nx,ny,step+1,s,sr,sc):
            return True
        s[nx][ny]=0
    return False
n=int(input())
r,c=map(int,input().split())
s=[[0]*n for _ in range(n)]
s[r][c]=1
print(['fail','success'][dfs(n,r,c,1,s,r,c)])
```















