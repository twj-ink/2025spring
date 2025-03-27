from collections import deque
n=int(input())
outer=deque()
inner=[deque() for _ in range(n)]
d={}
for i in range(n):
    s=list(input().split())
    for num in s:
        d[num]=i
while True:
    s=input().strip()
    if s=='STOP':
        break
    if s.startswith('ENQUEUE'):
        num=s.split()[1]
        if num in d:
            idx=d[num]
            inner[idx].append(num)
            if len(inner[idx])==1:
                outer.append(inner[idx])
        else:
            outer.append(deque([num]))
    else:
        if outer and outer[0]:
            out=outer[0].popleft()
            if not outer[0]:
                outer.popleft()
            print(out)











# inner=deque()
# outer=deque()
# inner.append(1)
# outer.append(inner)
# print(outer)
# inner.append(2)
# print(outer)
# outer[0].append(3)
# print(inner)