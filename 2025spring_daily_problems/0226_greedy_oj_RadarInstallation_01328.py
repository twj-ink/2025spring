case=0
while True:
    case+=1
    f=True
    n,d=map(int,input().split())
    if {n,d}=={0}:
        break
    xy=[]
    for _ in range(n):
        x,y=map(int,input().split())
        if y>d:
            f=False
        xy.append((x,y))
    if f:
        projection=[]
        for x,y in xy:
            left=x-(d**2-y**2)**0.5
            right=x+(d**2-y**2)**0.5
            projection.append((left,right))

        projection.sort(key=lambda x:x[1])
        num=1
        l,r=projection[0][0],projection[0][1]
        for i in range(1,n):
            if projection[i][0]<=r:
                continue
            else:
                r=projection[i][1]
                num+=1
    else:
        num=-1
    print(f'Case {case}: {num}')
    input()

