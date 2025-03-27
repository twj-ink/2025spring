from collections import defaultdict

def solve():
    n=int(input())
    s=list(map(int,input().split()))
    d={2:2,0:3,1:1,5:1,3:1}
    dd=defaultdict(int)
    for i in range(n):
        dd[s[i]]+=1
        if all(k in dd and d[k]<=dd[k] for k in d):
            return i+1
    return 0
def main():
    t=int(input())
    for _ in range(t):
        print(solve())

main()