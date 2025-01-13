n,m=map(int,input().split())
s=list(map(int,input().split()))
ans=[]
for _ in range(m):
    a,b=input().split()
    b=int(b)
    if a=='C':
        for i in range(n):
            s[i]=(s[i]+b)%65535
    else:
        cnt=0
        for i in s:
            if (i >> b) & 1:
                cnt+=1
        ans.append(cnt)
for i in ans:
    print(i)
