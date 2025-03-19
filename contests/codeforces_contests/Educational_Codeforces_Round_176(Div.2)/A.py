for _ in range(int(input())):
    n,k=map(int,input().split())
    if n & 1:
        n-=k
        if n%(k-1)==0:
            print(1+n//(k-1))
        else:
            print(2+n//(k-1))
    else:
        if n % (k - 1) == 0:
            print(n // (k - 1))
        else:
            print(1 + n // (k - 1))
