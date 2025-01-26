s=list(map(int,input().split()))
a=[i for i in range(1,6)]
c=0
for i in range(4):
    if s[i]!=a[i] and s[i+1]!=a[i+1]:
        c+=1
print('Yes' if c==1 else 'No')