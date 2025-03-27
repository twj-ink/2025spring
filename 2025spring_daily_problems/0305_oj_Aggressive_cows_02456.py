def canReach(s,mid):
    cnt=1
    curr=s[0]
    for i in range(1,len(s)):
        if mid<=s[i]-curr:
            cnt+=1
            curr=s[i]
        else:
            continue
    return cnt>=c

n,c=map(int,input().split())
s=[]
for _ in range(n):
    s.append(int(input()))

s.sort()
left,right=0,s[-1]-s[0]
while left<=right:
    mid=(left+right)//2
    if canReach(s,mid):
        left=mid+1
    else:
        right=mid-1
print(right)