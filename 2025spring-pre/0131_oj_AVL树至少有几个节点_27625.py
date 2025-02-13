# N(h)=1+N(h-1)+N(h-2)

def avl(n):
    if n==1:
        return 1
    if n==2:
        return 2
    a,b=1,2
    for _ in range(n-2):
        a,b=b,1+a+b
    return b

n=int(input())
print(avl(n))