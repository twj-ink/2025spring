### D ###
for _ in range(int(input())):
    n=int(input())
    s=list(map(int,input().split()))
    if s==sorted(s) or n==1:
        print('YES')
        continue
    if n>=2 and s[0]>s[1]:
        print('NO')
        continue
    for i in range(n-2):
        s[i+1]-=s[i]
        if s[i+1]>s[i+2]:
            print('NO')
            break
    else:
        print('YES')
