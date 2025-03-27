dx,dy=[-1,1,-2,2,-2,2,-1,1],[-2,-2,-1,-1,1,1,2,2]
def dfs(s,p,q,x,y,path):
    target=p*q
    s[x][y]=1
    xx=str(x+1)
    yy=chr(ord('A')+y)
    path+=(yy+xx)
    if len(path)==target*2:
        paths.append(path)
        return
    for i in range(8):
        nx,ny=x+dx[i],y+dy[i]
        if 0<=nx<p and 0<=ny<q and s[nx][ny]==0:
            s[nx][ny]=1
            dfs(s,p,q,nx,ny,path)
            s[nx][ny]=0

n=int(input())
for case in range(1,n+1):
    paths=[]
    p,q=map(int,input().split()) #1,2,3...,p;;;;a,b,c,,,,,,q
    s=[[0]*q for _ in range(p)]
    f=True
    for j in range(q):
        if not f:
            break
        for i in range(p):
            dfs(s,p,q,i,j,'')
            if paths:
                f=False
                break

    print(f'Scenario #{case}:')
    if not paths:
        print('impossible')
    else:
        paths.sort()
        print(paths[0])
    print()