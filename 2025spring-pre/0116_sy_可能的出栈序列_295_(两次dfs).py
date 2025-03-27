#用dfs两次，第一次是模拟入栈的操作，第二次是模拟出栈的操作
ans=[]
def dfs(n,push,stack,popped):
    global ans
    if not push and not stack:
        ans.append(popped[:])
        return
    if push:
        stack.append(push[0])
        dfs(n,push[1:],stack,popped)
        stack.pop()

    if stack:
        popped.append(stack.pop())
        dfs(n,push,stack,popped)
        stack.append(popped.pop())

n=int(input())
dfs(n,[i for i in range(1,n+1)],[],[])
ans.reverse()
for i in ans: print(*i)

#例如，给定一个集合，找到其所有子集，包含**选**与**不选**两种选择
#也可以使用两次dfs
def dfs(nums,path,idx,res):
    if idx==len(nums):
        res.append(path[:])
        return

    dfs(nums,path,idx+1,res)

    path.append(nums[idx])
    dfs(nums,path,idx+1,res)
    path.pop()

res=[]
dfs((1,2,3,4),[],0,res)
for i in res: print(i)