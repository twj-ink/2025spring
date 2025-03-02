n=int(input())
d={}
for _ in range(n):
    s=list(map(int,input().split()))
    d[_]=set(s[1:])

m=int(input())
for _ in range(m):
    l=list(map(int,input().split()))
    idx=l.index(1)
    start=d[idx].copy()
    for num in range(n):
        if l[num]==1:
            start&=d[num]
        elif l[num]==-1:
            start-=d[num]
    if start:
        print(*sorted(start))
    else:
        print('NOT FOUND')

#
# a,b={1,2,3},{1,2}
# print(a&b)
# print(a-b)