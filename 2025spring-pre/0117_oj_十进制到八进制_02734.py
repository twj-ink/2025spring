stack=[]
a=int(input())
while a:
    stack.append(a%8)
    a//=8
stack.reverse()
print(''.join(map(str,stack)))