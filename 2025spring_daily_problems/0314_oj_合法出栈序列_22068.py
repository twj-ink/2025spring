def check(s,curr):
    if len(curr)<n:
        return False
    i=0
    stack=[]
    for c in s:
        stack.append(c)
        while stack and i<n and stack[-1]==curr[i]:
            stack.pop()
            i+=1
    return i==n


s=input()
n=len(s)
while True:
    try:
        curr=input()
        print(['NO','YES'][check(s,curr)])
    except EOFError:
        break

