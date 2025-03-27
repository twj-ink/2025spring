# s=input()
# stack=[]
# ans=''
# for i in s:
#     if i in ('+','-'):
#         while stack:
#             ans+=stack.pop()
#         stack.append(i)
#     elif i.isdigit():
#         ans+=i
# while stack:
#     ans+=stack.pop()
# print(' '.join(ans))
##################中缀转后缀########
#遇到数字直接放入ans
#遇到操作符，将stack中优先级大于等于其的全部弹出放入ans，再进入stack
#遇到左括号，直接压入stack
#遇到右括号，不断弹出stack直到遇到左括号
###如果有括号，'('设置为最低级的0，')'要一直弹出直到遇到左括号
d={'+':1,'-':1,'*':2,'/':2} # '(':0
s=input()
stack=[]
ans=''
for i in s:
    if i.isdigit():
        ans+=i
    elif i in ('+','-','*','/'):
        while stack:
            if d[stack[-1]]>=d[i]:
                ans+=stack.pop()
            else:
                break
        stack.append(i)
while stack:
    ans+=stack.pop()
print(' '.join(ans))

#################后缀表达式求值
#特点：数字先出现，符号后出现
#遇到数字压入stack，遇到符号取出两个数字计算再压入
#注意：先取出的是右操作数
s=input()
stack=[]
for i in s:
    if i.isdigit():
        stack.append(i)
    elif i in ('+','-','*','/'):
        right=stack.pop()
        left=stack.pop()
        stack.append(eval(f'{left}{i}{right}'))
print(format(stack[0],'.2f'))

#################
#简单计算器--> 将中缀使用栈转化为后缀（注意运算符优先级，压入
#时要把优先级大于等于其的都弹出）
#然后使用栈模拟后缀的运算过程（注意先弹出的是右操作数）
#遇到数字直接放入ans
#遇到操作符，将stack中优先级大于等于其的全部弹出放入ans，
#再进入stack
###如果有括号，'('设置为最低级的0，')'要一直弹出直到遇到左括号
d={'+':1,'-':1,'*':2,'/':2}
s=input()
stack=[]
ans=''
for i in s:
    if i.isdigit():
        ans+=i
    elif i in ('+','-','*','/'):
        while stack:
            if d[stack[-1]]>=d[i]:
                ans+=stack.pop()
            else:
                break
        stack.append(i)
while stack:
    ans+=stack.pop()
#特点：数字先出现，符号后出现
#遇到数字压入stack，遇到符号取出两个数字计算再压入
#注意：先取出的是右操作数

stack=[]
for i in ans:
    if i.isdigit():
        stack.append(i)
    elif i in ('+','-','*','/'):
        right=stack.pop()
        left=stack.pop()
        stack.append(eval(f'{left}{i}{right}'))
print(format(stack[0],'.2f'))


###或者###
print(format(eval(input()),'.2f'))