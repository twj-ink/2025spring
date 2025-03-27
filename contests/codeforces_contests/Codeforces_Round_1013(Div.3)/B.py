def solve():
    n,x=map(int,input().split())
    s=list(map(int,input().split()))
    s.sort(reverse=True)
    ans=0
    curr_min=s[0]
    cnt=0
    for i in s:
        curr_min = min(curr_min,i)
        cnt+=1
        if curr_min*cnt>=x:
            ans+=1
            cnt=0
    return ans




def main():
    t = int(input())
    for _ in range(t):
        print(solve())


main()