# 图算法汇总



## 无向图/有向图的判环

### 一、 Undirected graph

#### 1. DFS + visited + parent(prev)

在建图之后操作，由于可能会存在多个不连通的独立的图，所以要对所有的节点进行dfs。

parent是指当前node的来源地，当遍历这个node的邻居时，考虑两种情况：

1. 不在visited中，说明是新的节点，继续dfs；
2. 在visited中，由于是无向图，有可能这个邻居实际上就是这个节点的来源parent（最近的那个把该节点延申出来的节点），此时不能说明有环；
所以判断一下这个邻居是不是parent，不是就说明有环了。

如果是矩阵类型的图，visited设置成二维布尔数组即可，然后parent或者说是prev设置为上一个位置的坐标即可；如果是一般的图，visited用一个set存遍历过的节点，parent是上一个节点。

下面给出的分别是**cpp的矩阵图代码**和**python的一般图代码**。

在第一段代码中，prev使用的是`i*m+j`，替代了pair<int,int> {i,j}，visited设置的是`vector<vector<char>> visited(n, vector<char>(m, false));`。

```cpp
// 题目url：http://dsaex.openjudge.cn/finalreview/C/

#include <bits/stdc++.h>
using namespace std;

using vvb = vector<vector<char>>;
int dx[4]={-1,0,0,1},dy[4]={0,1,-1,0};

bool has_cycle(const int& i, const int& j, int prev, const vector<string>& g,int n,int m,vvb& visited ) {
	visited[i][j]=true;
	for (int k = 0; k < 4; k++) {
		int nx = i+dx[k], ny = j+dy[k];
		if (0<=nx&&nx<n&&0<=ny&&ny<m&&g[nx][ny]==g[i][j]) {
			if (visited[nx][ny]==false) {
				if (has_cycle(nx,ny,i*m+j,g,n,m,visited)) {
					return true;
				}
			} else {
				if (nx*m+ny!=prev) {
					return true;
				}
			}
		}
	}
	return false;
}

void solve() {
	int n,m;
	cin >>n>>m;
	vector<string> g(n);
	for (int i = 0;i<n;i++) {
		cin >>g[i];
	}
	vvb visited(n,vector<char>(m,false));
	bool has=false;
	for (int i = 0;i<n;i++) {
		if (has) break;
		for (int j=0;j<m;j++) {
			if (has) break;
			if (visited[i][j]==false){
				if (has_cycle(i,j,-1,g,n,m,visited)) {
				has=true;
			}
		}
	}
}
	if (has) cout<<"Yes"<<endl;
	else cout<<"No"<<endl;
}

int main() {
    ios::sync_with_stdio(false);
	cin.tie(0);

	int t;
	cin >> t;
	while (t--) {
		solve();
	}

    return 0;
}
```

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
```

#### 2. Disjoint Set Union

与dfs不同，并查集则是在建图的过程中进行判断，在union的过程中如果发现两个节点有共同的根，说明已经有环了。

```python
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

### 二、 Directed graph

#### 1. DFS + visited + stack

与无向图不同的地方在于，无向图成环只能是当前节点与除了邻居节点之外的节点形成边（当然是dfs下来的节点）；而有向图
为了考虑dfs下来的节点，使用一个stack在记录当前的遍历到的节点，结束后再删除，防止误判。

```python
def has_cycle_directed_dfs(graph):
    visited = set()
    stack = set()
    
    def dfs(node):
        visited.add(node)
        stack.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in stack:
                return True
        stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    
    return False
```

```cpp
// url: http://cs101.openjudge.cn/2025sp_routine/09202/

#include <bits/stdc++.h>
using namespace std;

using umap = unordered_map<int, vector<int> >;
unordered_set<int> visited, stk;

bool has_cycle(int u, const umap& g) {
	visited.insert(u);
	stk.insert(u);

	// 如果该节点无出度，直接返回无环
	if (!g.count(u)) {
		stk.erase(u);
		return false;
	}

	for (int nei : g.at(u)) { 
		if (!visited.count(nei)) {
			if (has_cycle(nei, g)) {
				return true;
			}
		} else if (stk.count(nei)) { // 检查是否在递归栈中
			return true;
		}
	}
	stk.erase(u); // 从递归栈中移除
	return false;
}

void solve() {
	int n,m;
	cin >>n>>m;
	umap g;
	for (int i = 0;i<m;i++) {
		int u,v;
		cin >> u>>v;
		g[u].push_back(v);
	}

	bool has=false;
	visited.clear();
	stk.clear();

	for (int u = 1; u <= n; u++) { 
		if (!visited.count(u) && has_cycle(u, g)) {
			has = true;
			break;
		}
	}
	
	if (has) cout<<"Yes"<<endl;
	else cout<<"No"<<endl;
}

int main() {
    ios::sync_with_stdio(false);
	cin.tie(0);

	int t;
	cin >> t;
	while (t--) {
		solve();
	}

    return 0;
}
```


#### 2. Topological Sort

不断寻找入度为0的节点。

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
```

#### DFS + color

```python
from collections import defaultdict
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


## 有向图的强连通分量(Kosaraju算法/Tarjan算法)

强连通分量（SCC）：

在有向图中，如果两个顶点u和v满足：

从u可以到v，从v也可以到u。

那么u和v是强连通的。一个**强连通分量**是一个极大的（最大）子图，其中任意两个点都强连通。

### Kosaraju:两次dfs

第一次dfs获取递归栈stack，记录每个节点遍历的完成时间顺序；

然后将有向图反转；

第二次dfs不断弹出stack的节点获取各个强连通分量。

下面是cpp和python代码。

```cpp
#include <bits/stdc++.h>
using namespace std;

using Graph = unordered_map<int, vector<int>>;

void dfs1(int u, const Graph& graph, unordered_set<int>& visited, stack<int>& stk) {
    visited.insert(u);
    for (int v : graph[u]) {
        if (!visited.count(v)) {
            dfs1(v, graph, visited, stk);
        }
    }
    stk.push(u); // 在遍历完后压入栈
}

void dfs2(int u, const Graph& rev_graph, unordered_set<int>& visited, vector<int>& comp) {
    comp.push_back(u);
    visited.insert(u);
    if (rev_graph.count(u)) {
        for (int neighbor : rev_graph.at(u)) { // 用at，如果无键会报错，先count判断是否有键
            if (!visited.count(neighbor)) {
                dfs2(neighbor, rev_graph, visited, comp);
            }
        }
    }
}

auto kosaraju(const Graph& graph) {
    vector<vector<int>> comps;

    unordered_set<int> visited; // 一般图用set记录是否访问过
    stack<int> stk; // 记录第一次dfs的访问顺序

    // dfs1
    for (const auto& [u,_] : graph) {
        if (!visited.count(u)) {
            dfs1(u, graph, visited, stk);
        }
    }

    // reverse the graph
    Graph rev_graph;
    for (const auto& [u, neighbors] : graph) {
        for (int v : neighbors) {
            rev_graph[v].push_back(u);
        }
    }

    // dfs2
    visited.clear();
    while (!stk.empty()) {
        int u = stk.top(); stk.pop();
        if (!visited.count(u)) {
            vector<int> comp;
            dfs2(u, rev_graph, visited, comp);
            comps.push_back(comp);
        }
    }

    return comps;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    Graph graph;
    auto ans = kosaraju(graph);

    return 0;
}
```

```python
from collections import defaultdict

def dfs1(u, g, visited, stack):
    visited.add(u)
    for nei in g[u]:
        if nei not in visited:
            dfs1(nei, g, visited, stack)
    stack.append(u)

def dfs2(u, g, visited, comp):
    comp.append(u)
    visited.add(u)
    for nei in g[u]:
        if nei not in visited:
            dfs2(nei, g, visited, comp)
    


def kosaraju(graph):
    comps = []
    visited = set()
    stack = []

    for u,_ in graph.items():
        if u not in visited:
            dfs1(u,graph,visited,stack)
    
    rev_graph = defaultdict(list)
    for u,vs in graph.items():
        for v in vs:
            rev_graph[v].append(u)
    
    visited = set()
    while stack:
        u = stack.pop()
        comp = []
        if u not in visited:
            dfs2(u,rev_graph,visited,comp)
            comps.append(comp)
    
    return comps
```

### Tarjan:

////

## 最短路算法

### Dijkstra

解决单源非负权图最短路。写法与bfs很相近，因为当权重都为1时，该算法就退化为bfs了。每次取出的节点都是当前的最短路径的节点，然后用这个节点更新其相邻的节点。

用到的数据结构是堆，heap/priority_queue，保证每次取出最短路径，存入的是一个二元组(length, node)。

维护一个dist数组，dist[i]表示从源头到i节点的最短路。

Dijkstra 的特性保证：第一次从堆中取出的某个点，一定是从源点到它的最短路径已经确定，无需再次访问。

考虑优化：
1. 每次弹出当前最短路和节点之后、进行相邻节点循环之前，进行判断，查看当前弹出的distance是不是小于等于dist[node]，如果不是直接continue。这同时也保证了有环时能正常结束，取代了visited数组。
2. 如果只要找某个特定节点的最短路，那么在弹出当前最短路和节点之后，即可立即判断node==target，如果是则直接返回最短路。


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

例题：

#### lc-到达最后一个房间的最少时间II-3342

这是一个二维数组，标准的dijkstra，建立dist，维护一个heap，同时用上了两个优化，因为一定会到最后一个房间。省略了visited数组是因为第二处剪枝的优化。这里`new_d = max(d, moveTime[x][y]) + time`和`time = (i + j) % 2 + 1`值得学习，每步的额外time是1，2，1，2交错的，类似于象棋棋盘，(i+j)的奇偶性决定了time。

```python
class Solution:
    def minTimeToReach(self, moveTime: List[List[int]]) -> int:
        n,m = len(moveTime),len(moveTime[0])
        dist = [[float('inf')] * m for _ in range(n)]
        dist[0][0] = 0
        heap = [(0,0,0)]
        while True:
            d, i, j = heappop(heap)
            if i==n-1 and j==m-1:
                return d
            if d > dist[i][j]:
                continue
            
            time=(i+j)%2+1
            for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
                if 0<=x<n and 0<=y<m:
                    new_d = max(d, moveTime[x][y]) + time
                    if new_d < dist[x][y]:
                        dist[x][y] = new_d
                        heappush(heap,(new_d,x,y))
```


### Bellman-Ford

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

#### lc-网络延迟时间-743

即求所有最短路的最大值，要求把dist更新完全，下面给出了dijkstra、bellman、SPFA三种写法。

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        def bellman(times,n,k):
            dist = [float('inf')] * (n + 1)
            dist[k] = 0

            for _ in range(n - 1):
                clone = dist[:]
                for u, v, w in times:
                    if clone[u]!=float('inf') and clone[u] + w < dist[v]:
                        dist[v] = clone[u] + w
            return max(dist[1:]) if max(dist[1:])!=float('inf') else -1

        def SPFA(times,n,k):
            g = defaultdict(list)
            for u, v, w in times:
                g[u].append((v,w))
            dist = [float('inf')] * (n + 1)
            dist[k] = 0
            q = deque([k])

            for _ in range(n):
                clone = dist[:]
                for _ in range(len(q)):
                    u = q.popleft()
                    for v, w in g[u]:
                        if clone[u]!=float('inf') and clone[u] + w < dist[v]:
                            dist[v] = clone[u] + w
                            q.append(v)
            return max(dist[1:]) if max(dist[1:])!=float('inf') else -1
 
        def dijkstra(times,n,k):
            g = defaultdict(list)
            for u, v, w in times:
                g[u].append((v,w))

            dist = [float('inf')] * (n + 1)
            dist[k] = 0
            heap = [(0, k)]

            while heap:
                cur_len, node = heappop(heap)
                if cur_len > dist[node]:
                    continue
                
                for v, w in g[node]:
                    new_len = cur_len + w
                    if new_len < dist[v]:
                        dist[v] = new_len
                        heappush(heap, (new_len, v))
            
            return max(dist[1:]) if max(dist[1:])!=float('inf') else -1

        return dijkstra(times,n,k)
```

### Floyd-Warshall

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

## 加权无向连通图的最小生成树MST

### Prim

从点的角度出发。维护一个heap堆和visited数组，记录当前已经放入最小树节点的个数，一开始随便放入初始节点，接下来不断弹出节点，更新邻居节点，由于堆结构弹出的是最小边权，只要节点不在visited中就可以加入。

弹出的过程实际上就是选出 当前节点邻居的最短路径的过程。

建图的时候记得是无向图！

如果是邻接矩阵的形式，对于u的邻居直接遍历`for v in range(n):`即可，边权就是`g[u][v]`。

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

### Kruskal

从边的角度出发，一开始对边权排序，依次选择最小边，检测两点是否会形成环，使用并查集来确保。

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

## 关键路径

### 拓扑排序 + 关键路径长度

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