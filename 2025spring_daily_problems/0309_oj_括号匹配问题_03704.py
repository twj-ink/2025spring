while True:
    try:
        s=input()
        ans=[' ']*len(s)
        stack=[]
        for i,c in enumerate(s):
            # print(stack)
            if c=='(':
                stack.append((c,i))
            elif c==')':
                while stack and stack[-1][0]!='(':
                    stack.pop()
                if not stack:
                    ans[i]='?'
                    continue
                if stack[-1][0]=='(':
                    stack.pop()
            else:
                stack.append((c,i))
        for i,c in stack:
            if i=='(':
                ans[c]='$'
        # print(stack)
        print(s)
        print(''.join(ans))


    except EOFError:
        break