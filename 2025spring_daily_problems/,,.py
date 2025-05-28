# from collections import deque,defaultdict
# import sys
# sys.setrecursionlimit(1 << 30)
# for _ in range(int(input())):
#     n=int(input())
#     words=[input() for _ in range(n)]
#     g = defaultdict(lambda : defaultdict(deque))
#     indeg = defaultdict(int)
#     outdeg = defaultdict(int)
#     nodes = set()
#
#     for word in words:
#         s,e=word[0],word[-1]
#         g[s][e].append(word)
#         indeg[e]+=1
#         outdeg[s]+=1
#         nodes.add(s)
#         nodes.add(e)
#
#     for u in g:
#         for v in g[u]:
#             g[u][v] = deque(sorted(g[u][v]))
#
#
#     valid = True
#     start_node, end_node = [], []
#     for node in nodes:
#         ind,outd=indeg[node],outdeg[node]
#         if ind - outd == 1:
#             end_node.append(node)
#         elif outd - ind == 1:
#             start_node.append(node)
#         elif ind != outd:
#             valid = False
#
#     if len(start_node) > 1 or len(end_node) > 1:
#         valid = False
#
#     if not valid:
#         print('***')
#         continue
#
#     if len(start_node) == 1:
#         start = start_node[0]
#     else: # Euler circuit
#         circle = [node for node in outdeg if outdeg[node] > 0]
#         if not circle:
#             print('***')
#             continue
#         start = min(circle)
#
#     path = []
#     def dfs(u, g, path):
#         nextv = sorted(v for v in g[u])
#         for v in nextv:
#             while g[u][v]:
#                 word = g[u][v].popleft()
#                 dfs(v, g, path)
#                 path.append(word)
#
#     dfs(start, g, path)
#     if len(path) != len(words) or not path:
#         print('***')
#         continue
#
#     path.reverse()
#     print('.'.join(path))
n=int(input())
cnt=0
stack=[]
p=1
for _ in range(2*n):
    cmd,*x=input().split()
    if cmd=='add':
        stack.append(int(x[0]))
    else:
        if stack[-1]!=p:
            stack.sort(reverse=True)
            cnt+=1
        stack.pop()
        p+=1
print(cnt)