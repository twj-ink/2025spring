from collections import defaultdict
n=int(input())
d=defaultdict(set)

for idx in range(1,n+1):
    s=list(input().split())
    for word in s[1:]:
        if idx not in d[word]:
            d[word].add(idx)

m=int(input())
for _ in range(m):
    w=input()
    if w in d:
        print(' '.join(map(str,sorted(d[w]))))
    else:
        print('NOT FOUND')