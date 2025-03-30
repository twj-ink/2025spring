#pylint:skip-file
def solve(a,i):
    global ans
    if a==1:
        ans+=1
        return 

    for j in range(i,a+1):
        if a%j==0:
            solve(a//j,j)



for _ in range(int(input())):
    a=int(input())
    ans=0
    solve(a,2)
    print(ans)
