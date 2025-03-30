def kadane(s):
    curr_max=total_max=s[0]
    for i in range(1,len(s)):
        curr_max=max(curr_max+s[i],s[i])
        total_max=max(total_max,curr_max)
    return total_max

def max_mat(mat):
    max_sum=-float('inf')
    n,m=len(mat),len(mat[0])
    for top in range(n):
        col_sum=[0]*m
        for bot in range(top,n):
            for j in range(m):
                col_sum[j] += mat[bot][j]
            max_sum=max(max_sum,kadane(col_sum))
    return max_sum

n=int(input())
nums=[]
while len(nums)<n**2:
    nums.extend(input().split())
mat=[list(map(int,nums[i*n:(i+1)*n])) for i in range(n)]
print(max_mat(mat))