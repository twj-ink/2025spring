from math import ceil
def solve():
    n=int(input())
    s=[i for i in range(1,n+1)]
    if not n&1:
        return -1
    ans=[0]*n
    mid=ceil(n/2)
    ans[mid-1]=mid
    i,j=mid-1,mid-1
    for k in range(n-1):
        i+=2
        j+=1
        ans[j%n]=s[i%n]
    return ans


def main():
    t = int(input())
    for _ in range(t):
        ans=solve()
        if ans==-1:
            print(ans)
        else:
            print(*ans)


main()