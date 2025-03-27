n,m=map(int,input().split())
s=[]
left=top=98899999;right=bottom=-1
for i in range(n):
    l=input()
    for j in range(m):
        if l[j]=='#':
            left=min(left,j)
            right=max(right,j)
            top=min(top,i)
            bottom=max(bottom,i)
    s.append(l)
# print(left,right,top,bottom)
f=True
for i in range(top,bottom+1):
    if f:
        for j in range(left,right+1):
            if s[i][j]=='.':
                f=False
                break
print('Yes' if f else 'No')
