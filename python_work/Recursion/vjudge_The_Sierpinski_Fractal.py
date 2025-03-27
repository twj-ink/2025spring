while True:
    n=int(input())
    if n==0:
        break
    s=[[' ']*(2**(n+1)) for _ in range(2**n)]
    a,b=len(s)-1,0
    def dr(s,n,x,y):
        if n==1:
            s[x][y:y+4]="/__\\"
            s[x-1][y+1:y+3]='/\\'
            return
        c,d=x,y+2**n
        e,f=x-2**(n-1),y+2**(n-1)
        dr(s,n-1,x,y)
        dr(s,n-1,c,d)
        dr(s,n-1,e,f)

    dr(s,n,a,b)
    for i in s:
        l=''.join(i)
        l.rstrip()
        print(l)
    print()