for _ in range(int(input())):
    n,k=map(int,input().split())
    stack=[]
    cnt=0
    s=str(n)
    for i in range(len(s)):
        while stack and s[stack[-1]]>s[i] and cnt<k:
            cnt+=1
            stack.pop()
        stack.append(i)
    if cnt<k:
        rest=k-cnt
        for _ in range(rest):
            stack.pop()
    ans=''
    for i in stack:
        ans+=s[i]
    print(ans)