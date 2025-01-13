#很清晰的Tutorial
'''
The naive solution of writing out a linear system and solving them will take O((n+m)3) time, which is too slow,
so we will need a faster algorithm.

We begin by selecting a target sum S for each row and column. If we calculate the sum of all numbers in the completed
grid, summing over rows gives a total of S⋅n while summing over columns gives a total of S⋅m. Therefore, in order for our
choice of S to be possible, we require S⋅n=S⋅m, and since it is possible for n≠m, we will pick S=0 for our choice to be
possible in all cases of n,m. Notice that all choices S≠0 will fail on n≠m, as the condition S⋅n=S⋅m no longer holds. As
such, S=0 is the only one that will work in all cases.

Now, we aim to make each row and column sum to S. The crux of the problem is the following observation:

Denote x1,x2,…,xn+m−1 to be the variables along the path. Let's say variables x1,…,xi−1 have their values set for some
1≤i<n+m−1. Then, either the row or column corresponding to variable xi has all of its values set besides xi, and
therefore we may determine exactly one possible value of xi to make its row or column sum to 0.

The proof of this claim is simple. At variable xi, we look at the corresponding path move si. If si=R, then the path
will never revisit the column of variable xi, and its column will have no remaining unset variables since x1,…,xi−1 are
already set. Likewise, if si=D, then the path will never revisit the row of variable xi, which can then be used to determine the value of xi.

Repeating this process will cause every row and column except for row n and column m to have a sum of zero, with xn+m−1
being the final variable. However, we will show that we can use either the row or column to determine it, and it will
give a sum of zero for both row n and column m. WLOG we use row n. Indeed, if the sum of all rows and columns except for
column m are zero, we know that the sum of all entries of the grid is zero by summing over rows. However, we may then
subtract all columns except column m from this total to arrive at the conclusion that column m also has zero sum.
Therefore, we may determine the value of xn+m−1 using either its row or column to finish our construction, giving a solution in O(n⋅m).
'''
for _ in range(int(input())):
    n,m=map(int,input().split())
    s=input()
    l=n+m-2
    a=[]
    for _ in range(n):
        a.append(list(map(int,input().split())))

    x,y=0,0
    for i in range(l):
        if s[i]=='D':
            curr=sum(a[x][j] for j in range(m))
            a[x][y]=-curr
            x+=1
        else:
            curr=sum(a[j][y] for j in range(n))
            a[x][y]=-curr
            y+=1

    a[-1][-1]=-sum(a[-1][j] for j in range(m))

    for i in a:
        print(*i)
