#pylint:skip-file
def dfs(s,k,i,col,curr):
    global ans
    if i<=n and curr==k:
        ans+=1
        return
    if (i==n and curr<k) or (k-curr>n-i):
        return
    for j in range(n):
        if s[i][j]=='#' and not col[j]:
            col[j]=True
            dfs(s,k,i+1,col,curr+1)
            col[j]=False
    dfs(s,k,i+1,col,curr)

while True:
    n,k=map(int,input().split())
    if {n,k}=={-1}:
        break

    s=[input() for _ in range(n)]
    ans=0
    dfs(s,k,0,[False]*n,0)
    print(ans)