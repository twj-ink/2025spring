for _ in range(int(input())):
    n,k=map(int,input().split())
    s=list(map(int,input().split()))
    a=sorted(s,reverse=True)
    if k>=2:
        print(sum(a[:k+1]))
    else:
        p,q=a[0],a[1]
        if p in (s[0],s[-1]) or q in (s[0],s[-1]):
            print(sum(a[:k+1]))
        else:
            print(p+s[0] if s[0]>s[-1] else p+s[-1])

