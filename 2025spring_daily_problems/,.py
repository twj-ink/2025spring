def solve():
    n=int(input())
    s=list(map(int,input().split()))
    dp=[[0]*n for _ in range(n)]
    for r in range(2,n):
        for l in range(r-2,-1,-1):
            for i in range(l+1,r):
                dp[l][r]=max(dp[l][r],dp[l+1][i-1]+dp[i+1][r-1]+s[l]*s[i]*s[r],dp[l][i]+dp[i+1][r])
            for i in range(l,r):
                dp[l][r]=max(dp[l][r],dp[l][i]+dp[i+1][r])
    print(dp[0][-1])


def main():
    t=int(input())
    for _ in range(t):
        solve()

main()