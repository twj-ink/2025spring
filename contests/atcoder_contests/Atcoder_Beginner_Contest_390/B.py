from decimal import Decimal
n=int(input())
s=list(map(int,input().split()))
f=True
if n==2:
    print('Yes')
else:
    # r=s[2]*s[0]
    for i in range(2,n):
        if s[i-2]*s[i]!=s[i-1]*s[i-1]:
            f=False
    print('Yes' if f else 'No')
