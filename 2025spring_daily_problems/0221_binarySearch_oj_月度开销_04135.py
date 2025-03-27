def check(mid,s,m):
    curr=0
    cnt=1
    for i in s:
        if curr+i>mid:
            cnt+=1
            curr=i
        else:
            curr+=i
    return cnt<=m

def solve():
    n,m=map(int,input().split())
    s=[]
    for _ in range(n):
        s.append(int(input()))
    left=max(s)
    right=sum(s)
    while left<=right:
        mid=(left+right)//2
        if not check(mid,s,m):
            left=mid+1
        else:
            right=mid-1
    return left

print(solve())
