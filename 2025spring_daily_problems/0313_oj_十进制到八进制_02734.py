a=int(input())
stack=[]
while a>=1:
    stack.append(a%8)
    a//=8
print(''.join(map(str,stack[::-1])))