a,b,k=map(int,input().split())
cnt=0
s=[[0]*b for _ in range(a)]
for _ in range(k):
    x,y,r,t=map(int,input().split())
    cnt+=(t==1)
    r//=2
    for i in range(max(0,x-1-r),min(a-1+1,x-1+r+1)):
        for j in range(max(0,y-1-r),min(b-1+1,y-1+r+1)):
            s[i][j]+=1 if t==1 else -1

ans=sum(1 for i in range(a) for j in range(b) if s[i][j]==cnt)
print(ans)