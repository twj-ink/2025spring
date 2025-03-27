for _ in range(int(input())):
    s=list(input().split())
    stack=[]
    for i in s:
        if i not in ('+','-','*','/'):
            stack.append(i)
        else:
            right=stack.pop()
            left=stack.pop()
            stack.append(eval(f'{left}{i}{right}'))
    print(format(stack[0],'.2f'))