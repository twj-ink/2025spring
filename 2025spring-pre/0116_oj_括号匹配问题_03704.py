while True:
    try:
        s=input()
        mark=[' ']*len(s)
        stack=[]
        right=[]
        for i in range(len(s)):
            if s[i]=='(':
                stack.append((i,'$'))
            elif s[i]==')':
                if stack:
                    stack.pop()
                else:
                    right.append((i,'?'))
        stack.extend(right)
        for i in stack:
            mark[i[0]]=i[1]
        print(s)
        print(''.join(mark))

    except EOFError:
        break