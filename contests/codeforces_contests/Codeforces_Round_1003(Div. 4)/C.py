for _ in range(int(input())):
    n,m=map(int,input().split())
    a=list(map(int,input().split()))
    b=list(map(int,input().split()))
    b.sort()
    s=[]

    curr=-float('inf')
    for i in range(n):
        if s[i][0]>=curr:
            curr=s[i][0]
            continue
        elif s[i][1]>=curr:
            curr=s[i][1]
            continue
        else:
            # f=False
            print('NO')
            break
    else:
        print('YES')
