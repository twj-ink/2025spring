class DirFile:
    def __init__(self,val=None,dir=None,file=None):
        self.val=val
        self.dir=dir if dir is not None else []
        self.file=file if file is not None else []

def getDirFile(s):
    root=DirFile('ROOT')
    stack=[]
    stack.append(root)
    for i in s:
        if i[0]=='f':
            stack[-1].file.append(DirFile(i))
        elif i[0]=='d':
            newDir=DirFile(i)
            stack[-1].dir.append(newDir)
            stack.append(newDir)
        elif i==']':
            stack.pop()
    return stack[0] # 根节点ROOT

def show(plus,root,ans):
    ans.append(plus+root.val)
    if not root.dir and not root.file:
        return
    if root.dir:
        for d in root.dir:
            show('|     '+plus,d,ans)
    if root.file:
        root.file.sort(key=lambda x:x.val)
        for f in root.file:
            show(plus,f,ans)
    return ans


cnt=0
s=[]
while True:
    l=input()
    if l=='#':
        break
    if l=='*':
        cnt+=1
        root=getDirFile(s)
        print(f'DATA SET {cnt}:')
        ans=[]
        show('',root,ans)
        for i in ans: print(i)
        s=[]
        print()
    s.append(l)
