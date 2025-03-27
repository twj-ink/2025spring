def canreach(mid,n,m,k):
    line=mid*(m//(mid+1))+(m%(mid+1))
    return line*n>=k

def solve():
    n,m,k=map(int,input().split())
    l,r=0,m
    while l<=r:
        mid=(l+r)//2
        if canreach(mid,n,m,k):
            r=mid-1
        else:
            l=mid+1
    return l



def main():
    t=int(input())
    for _ in range(t):
        print(solve())

main()