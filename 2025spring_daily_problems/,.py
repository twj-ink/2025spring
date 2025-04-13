from collections import Counter,defaultdict,deque
from math import ceil,floor,atan
from functools import lru_cache
from typing import List,Optional

n,l=map(int,input().split())
s=[]
for _ in range(n):
    x,y,z=map(int,input().split())
    if x==0 and y>=0: d=0
    elif x>0 and y>0: d=atan(x/y)
    elif x>0 and y==0: d=90
    elif x>0 and y<0: d=atan((-y)/x)+90
    elif x==0 and y<0: d=180
    elif x<0 and y<0: d=atan((-x)/(-y))+180
    elif x<0 and y==0: d=270
    else: d=atan(y/(-x))+270
    s.append((d,0.5**((x**2)+(y**2)),x,y,z))

idx=sorted(range(n),key=lambda i:(i[0],i[1]))
i=-1
ans=[-1]*n
succ=False
for _ in range(n):
    if succ: break
    for i in range(n):
        curr_d=s[idx[i]][0]
        while l>=s[idx[i]][1]:
            l+=s[idx[i]][1]