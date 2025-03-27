#递归  TLE
n=int(input())
cnt=0
def dfs(n,nums,stack,popped):
    global cnt
    if not nums and not stack:
        cnt+=1
        return
    if nums:
        stack.append(nums[0])
        dfs(n,nums[1:],stack,popped)
        stack.pop()

    if stack:
        popped.append(stack.pop())
        dfs(n,nums,stack,popped)
        stack.append(popped.pop())
dfs(n,[i for i in range(1,n+1)],[],[])
print(cnt)

#DP
#dp[i][j]表示还有i个元素未入栈，栈中已有j个元素时的合法序列数
#则此时可以选择让i个元素中的一个入栈，即dp[i-1][j+1]
#或者选择让j个元素中的一个出栈，即dp[i][j-1]
#则当i>0,j>0时，dp[i][j]=dp[i-1][j+1]+dp[i][j-1]
#最后取的是dp[0][
n=int(input())
dp=[[0]*(n+1) for _ in range(n+1)]
