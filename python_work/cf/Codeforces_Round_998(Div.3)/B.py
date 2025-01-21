for _ in range(int(input())):
    n,m=map(int,input().split())
    s=[]
    for _ in range(n):
        s.append((_+1,(sorted(list(map(int,input().split()))))))
        s.sort(key=lambda x:x[1])
    curr=0
    f=True
    for j in range(m):
        if f:
            for i in range(n):
                if s[i][1][j]!=curr:
                    f=False
                    break
                curr+=1
    ans=[]
    for i in range(n):
        ans.append(s[i][0])
    print(' '.join(map(str,ans)) if f else -1)
