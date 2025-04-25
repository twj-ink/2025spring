### 无向图的连通数目
# 建图，每个点有访问/未访问两种状态，dfs的作用的设置点已访问，
# 对**每一个点**进行遍历，如果未访问cnt加1

def unordered_component(g):
    def dfs(i,g):
        visited[i]=True
        for next in g[i]:
            if not visited[next]:
                dfs(next,g)

    cnt=0
    visited=[False]*len(g)
    for vertex in g:
        if not visited[vertex]:
            cnt+=1
            dfs(vertex, g)

    return cnt

# 判断是否只有一个部分，只需要对一个节点dfs，然后判断visited是不是全为True

### 有向图判环
# 建图，每个点有未访问0 / 正在访问1 / 已访问2 三种状态，dfs的过程中遇到节点设为1，结束后设为2，如果
# dfs的时候发现遇到了1，说明有环；也要对每一个点遍历

def ordered_has_cycle(g):
    visited = [0]*len(g)
    def dfs(node,g) -> bool :
        if visited[node] == 1:
            return True # has cycle
        if visited[node] == 2:
            return False

        visited[node] = 1
        for next in g[node]:
            if dfs(next,g): # 如果在接下来的节点中出现了环
                return True

        visited[node] = 2
        return False

    for vertex in g:
        if not visited[vertex]:
            if dfs(vertex,g):
                return True

    return False



