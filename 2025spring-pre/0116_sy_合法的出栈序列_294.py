def solve():
    n=int(input())
    s=list(map(int,input().split()))
    stack=[]
    cur=1

    for i in s:
    #cur模拟即将入栈的元素
    #stack[-1]表示即将出栈的元素
    #当栈为空；或即将出栈的元素与num不相等时，压入栈中
        while cur<=i and (not stack or stack[-1]!=i):
            stack.append(cur)
            cur+=1
        #否则就出栈
        if stack and stack[-1]==i:
            stack.pop()
        else:
            return 'No'
    return 'Yes'

print(solve())