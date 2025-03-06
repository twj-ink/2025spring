map={'M':10**6, 'B':10**9}
from collections import defaultdict
n=int(input())
d=defaultdict(list)
for _ in range(n):
    name,num=input().split('-')
    d[name].append(num)

s=[]
for k,v in d.items():
    s.append((k,v))
# print(s)
s.sort(key=lambda x:x[0])
for k,v in s:
    print(k+': ',end='')
    v.sort(key=lambda x:float(x[:-1])*map[x[-1]])
    print(', '.join(v))