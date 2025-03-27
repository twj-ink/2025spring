while True:
    m,n=map(int,input().split())
    if {m,n}=={0}:
        break
    ans=0
    i=1
    while m*(2**i)<=n:
        ans+=2**(i-1)
        i+=1
    base=m*2**(i-1)
    right=base+2**(i-1)-1
    if n>right:
        ans+=2**(i-1)
    else:
        ans+=n-base+1
    print(ans)