# Dijkstra 算法

解决 **单源最短路问题**，其写法与正常的bfs很相似，基本写法就是把bfs中的双端队列deque换成最小堆heapq，此外再增加一个dist的距离数组来维护从起点
到各个节点的最短距离，heap中常常存的是(curr_dist, x, y)。

例题：

### oj-兔子与樱花-05443

http://cs101.openjudge.cn/2025sp_routine/05443/

这道题是要自行建立节点和边的图的问题，在进行算法之前需要一些对输入数据的处理。在该题中，各个地名为节点，距离为边的权值，
可以借助前（p+1）行的输入建立地点到索引的键值对，方便在dist数组中查询地点的索引。

由于答案中需要输出从起点到终点的路径，在算法进行的过程中维护一个prev数组，键值对为当前地点和先前地点，每次dist更新时同时把这个字典也更新一次。

```python
from heapq import heappop,heappush
from collections import defaultdict

def dijkstra(g,name_id,s,e):
    q=[]
    heappush(q,(s,0))
    dist=[float('inf')]*len(g)
    dist[name_id[s]]=0
    prev={'s':(None,None)}
    while q:
        for _ in range(len(q)):
            curr_node,curr_dist = heappop(q)
            # if curr_node==e:
            #     return get_path(s,e,prev)
            for next,w in g[curr_node]:
                if curr_dist+w>dist[name_id[next]]:
                    continue
                dist[name_id[next]]=min(curr_dist+w, dist[name_id[next]])
                heappush(q,(next,dist[name_id[next]]))
                prev[next]=(curr_node,w)
    return get_path(s,e,prev)

def get_path(s,e,prev):
    ans=[]
    while e!=s:
        ans.append(e)
        e,w=prev[e]
        ans.append(f'({w})')
    ans.append(e)
    ans.reverse()
    return ans

p=int(input())
g=defaultdict(list)
name_id={}
for id in range(p):
    name=input()
    name_id[name]=id

q=int(input())
for _ in range(q):
    a,b,w=input().split()
    w=int(w)
    g[a].append((b,w))
    g[b].append((a,w))

n=int(input())
for _ in range(n):
    s,e=input().split()
    path=dijkstra(g,name_id,s,e)
    print('->'.join(path))
```

### lc-接雨水II-407

https://leetcode.cn/problems/trapping-rain-water-ii/

在一维和二维接雨水中均采用单调栈，三维可以考虑：
由于最外层势必不会接到雨水，能接到的只能是内部的某些点，那就计算从每一个点(x,y)到最外层边界的所有可能路径中的 **能遇到的最高高度的最小值**，
类似于木桶的模型，找四周木桶壁最低的一个，与桶底相减就是能盛水的最大高度。

正向来看，是单源（从(x,y)开始）最短路径，为了方便模拟，可以反过来，初始时将四周的高度放入堆中，然后对四周的各个点开始向内扩散，每次遇到比当前高度低的
点时便可以更新答案。

同时维护一个vis数组来避免陷入循环（而在兔子与樱花中由于建图不存在环所以无需考虑重复遍历）。

```python
from typing import List
from heapq import heappush,heappop
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        n,m=len(heightMap),len(heightMap[0])
        heap=[]
        vis=[[False]*m for _ in range(n)]
        for j in range(m):
            heappush(heap, (heightMap[0][j],0,j))
            heappush(heap, (heightMap[-1][j],n-1,j))
            vis[0][j] = vis[n-1][j] = True
        for i in range(1,n-1):
            heappush(heap, (heightMap[i][0],i,0))
            heappush(heap, (heightMap[i][-1],i,m-1))
            vis[i][0] = vis[i][m-1] = True
        ans=0
        while heap:
            h,x,y=heappop(heap)
            for dx,dy in (0,-1),(0,1),(-1,0),(1,0):
                nx,ny=x+dx,y+dy
                if 0<=nx<n and 0<=ny<m and vis[nx][ny]==False:
                    if h>heightMap[nx][ny]: # 只有低了才能接到雨水
                        ans+=h-heightMap[nx][ny]
                    vis[nx][ny]=True # 保证每个点都能遍历到
                    heappush(heap, (max(h, heightMap[nx][ny]), nx, ny))
        return ans
```
