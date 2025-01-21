for _ in range(int(input())):
    s=input()
    stack=[]
    ans=[]
    d={'(':5,'+':1,'-':1,'*':2,'/':2}
    i=0
    while i<len(s):
        if s[i]=='(':
            stack.append('(')
            i+=1
        elif s[i]==')':
            while stack[-1]!='(':
                ans.append(stack.pop())
            stack.pop()
            i+=1
        elif s[i] in ('+','-','*','/'):
            while stack and d[stack[-1]]>=d[s[i]]:
                ans.append(stack.pop())
            stack.append(s[i])
            i+=1
        else:
            cur=''
            while i<len(s) and s[i] not in ['+','-','*','/','(',')']:
                cur+=s[i]
                i+=1
            ans.append(cur)
    while stack:
        ans.append(stack.pop())
    print(' '.join(ans))

