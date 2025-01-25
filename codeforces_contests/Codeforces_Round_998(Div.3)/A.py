### Codeforces Round 998(Div.3) ###
### A ###
for _ in range(int(input())):
    a,b,c,d=map(int,input().split())
    s=set()
    s.add(a+b)
    s.add(c-b)
    s.add(d-c)
    print(4-len(s))

