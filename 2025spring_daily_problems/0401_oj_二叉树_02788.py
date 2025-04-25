while True:
    m,n=map(int,input().split())
    if {m,n}=={0}:
        break
    ans=0
    i=1
    while m*(2**i)<=n:
        ans+=(2**(i-1))
        i+=1
    left=m*(2**(i-1))
    half=2**(i-1)
    if n>left+half-1:
        ans+=half
    else:
        ans+=n-left+1
    print(ans)
