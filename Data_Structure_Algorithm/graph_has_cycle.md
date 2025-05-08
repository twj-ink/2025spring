## Undirected graph

### 1. DFS + visited + parent(prev)

在建图之后操作，由于可能会存在多个不连通的独立的图，所以要对所有的节点进行dfs。

parent是指当前node的来源地，当遍历这个node的邻居时，考虑两种情况：

1. 不在visited中，说明是新的节点，继续dfs；
2. 在visited中，由于是无向图，有可能这个邻居实际上就是这个节点的来源parent（最近的那个把该节点延申出来的节点），此时不能说明有环；
所以判断一下这个邻居是不是parent，不是就说明有环了。

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

### 2. Disjoint Set Union

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

## Directed graph

### 1. DFS + visited + stack

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

### 2. Topological Sort

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

### DFS + color

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