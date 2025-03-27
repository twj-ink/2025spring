from math import ceil,floor

for _ in range(int(input())):
    x,n,m=map(int,input().split())
    minv,maxv,maxv1=x,x,x
    kn,km=n,m
    knn,kmm=n,m
    for _ in range(n+m):
        if minv==0:
            break
        if minv & 1:
            if knn:
                minv//=2
                knn-=1
            else:
                minv=ceil(minv/2)
                kmm-=1
        else:
            if kmm:
                minv//=2
                kmm-=1
            else:
                minv//=2
                knn-=1


    for _ in range(n+m):
        # print(maxv)
        if maxv==0:
            break
        if maxv & 1:
            if km>0:
                maxv=ceil(maxv/2)
                km-=1
                if maxv==1 and kn:
                    maxv=0
                    break
                elif maxv==1 and not kn:
                    maxv=1
                    break
            else:
                maxv//=2
                kn-=1
        else:
            if kn>0:
                maxv//=2
                kn-=1
            else:
                maxv//=2
                km-=1

    for i in range(n):
        maxv1//=2
        if maxv1==0:
            break
    for i in range(m):
        maxv1=ceil(maxv1/2)
        if maxv1==0:
            break
        if maxv1==1:
            break
    maxv=max(maxv,maxv1)

    print(minv,maxv)

