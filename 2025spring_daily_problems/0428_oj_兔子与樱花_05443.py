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