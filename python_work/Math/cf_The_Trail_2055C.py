for _ in range(int(input())):
    n,m=map(int,input().split())
    s=input()
    l=n+m-2
    a=[]
    for _ in range(n):
        a.append(list(map(int,input().split())))

    x,y=0,0
    for i in range(l):
        if s[i]=='D':
            curr=sum(a[x][j] for j in range(m))
            a[x][y]=-curr
            x+=1
        else:
            curr=sum(a[j][y] for j in range(n))
            a[x][y]=-curr
            y+=1

    a[-1][-1]=-sum(a[-1][j] for j in range(m))

    for i in a:
        print(*i)
