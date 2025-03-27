an,am=map(int,input().split())
a=[]
for _ in range(an): a.append(list(map(int,input().split())))
bn,bm=map(int,input().split())
b=[]
for _ in range(bn): b.append(list(map(int,input().split())))
cn,cm=map(int,input().split())
c=[]
for _ in range(cn): c.append(list(map(int,input().split())))
if am!=bn or an!=cn or bm!=cm:
    print('Error!')
else:
    for i in range(an):
        for j in range(bm):
            c[i][j]+=sum(a[i][k]*b[k][j] for k in range(am))
    for i in c:
        print(*i)