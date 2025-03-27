### C ###
for _ in range(int(input())):
    n,k=map(int,input().split())
    s=sorted(list(map(int,input().split())))
    i,j=0,n-1
    cnt=0
    while i<j:
        if s[i]+s[j]<k:
            i+=1
        elif s[i]+s[j]>k:
            j-=1
        else:
            i+=1
            j-=1
            cnt+=1
    print(cnt)