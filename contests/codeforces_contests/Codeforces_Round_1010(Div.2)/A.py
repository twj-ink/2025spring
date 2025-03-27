for _ in range(int(input())):
    n,m=map(int,input().split())
    ans=0
    cnt1=0
    s=[]
    for i in range(n):
        l=input()
        cnt1+=(1 if l.count('1')&1 else 0)
        s.append(l)
    ans=max(ans,cnt1)
    cnt1=0
    for j in range(m):
        cnt1+=(1 if (sum(1 for i in range(n) if s[i][j]=='1'))&1 else 0)
    ans=max(ans,cnt1)
    print(ans)