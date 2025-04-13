from collections import defaultdict

def dfs(root):
    curr=[root]+nodes[root]
    curr.sort()
    for num in curr:
        if num==root:
            print(root)
        else:
            dfs(num)

nodes=defaultdict(list)
all,childs=set(),set()
n=int(input())
for _ in range(n):
    root,*child=list(map(int,input().split()))
    nodes[root].extend(child)
    all.update([root]+child)
    childs.update(child)
root=list(all-childs)[0]

dfs(root)
