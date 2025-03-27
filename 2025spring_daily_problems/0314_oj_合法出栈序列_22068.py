def solve(s,l):
    stack=[]
    curr=0
    n=len(l)
    for i in range(n):
        while curr<=i and (not stack or s[stack[-1]]!=s[i]):
            stack.append(curr)
            curr+=1
        if stack and s[stack[-1]]==s[i]:
            stack.pop()
        else:
            return 'No'
    return "Yes"
s=input()
while True:
    try:
        l=input()
        print(solve(s,l))





    except EOFError:
        break