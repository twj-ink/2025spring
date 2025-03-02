# def binary():
#     l,r=0,L//(n-m+1) #最多是r的距离
#     while l<=r:
#         mid=(l+r)//(2)
#         if check(m,mid):
#             #如果可以做到这个距离
#             l=mid+1
#         else:
#             #如果距离太大
#             r=mid-1
#     return r


def check(m,mid):
    cnt=0
    curr=s[0]
    for i in range(1,n+2):
        if cnt>m:
            return False
        if s[i]-curr<mid:
            cnt+=1
        else:
            curr=s[i]
    if cnt>m:
        return False
    return True

L,n,m=map(int,input().split())
s=[0]
for _ in range(n):
    s.append(int(input()))
s.append(L)

left,right=1,L//(n-m+1)
while left<right:
    mid=left+(right-left)//2
    if not check(m,mid):
        right=mid
    else:
        left=mid+1
print(left-1)