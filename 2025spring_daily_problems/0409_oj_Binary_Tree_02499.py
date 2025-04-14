'''
2 11
3 21   12
4 5 31  23  32  13
5 7 8 41  34  53  25  52  35  43  14
a,b -- a,b-a||a-b,b
3,4--3,1 r--2,1 l--1,1 l
'''

def cal(a,b):
    l,r=0,0
    while a!=1 or b!=1:
        # print((l,r))
        if a==1: r+=b-1; break
        if b==1: l+=a-1; break
        if a<b:
            r+=b//a
            b%=a
        else:
            l+=a//b
            a%=b
    return l,r

n=int(input())
for _ in range(n):
    a,b=map(int,input().split())
    l,r=cal(a,b)
    print(f'Scenario #{_+1}:')
    print(l,r)
    print()