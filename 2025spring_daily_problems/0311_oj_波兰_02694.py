def cal(s,i):
    op=s[i]
    if op in '+-*/':
        left,i=cal(s,i+1)
        right,i=cal(s,i)
        return eval(f'{left}{op}{right}'),i
    elif op[0].isdigit():
        return op,i+1

s=list(input().split())
print(f'{cal(s,0)[0]:.6f}')

