#pylint:skip-file
class dirfile:
    def __init__(self,val,files=None,dirs=None):
        self.val=val
        self.files=files if files is not None else []
        self.dirs=dirs if dirs is not None else []

    def __lt__(self, other):
        return self.val<other.val

def build(s):
    root=dirfile('ROOT')
    stack=[]
    stack.append(root)
    for i in s:
        if i[0]=='f':
            stack[-1].files.append(dirfile(i))
        elif i[0]=='d':
            new_dir=dirfile(i)
            stack[-1].dirs.append(new_dir)
            stack.append(new_dir)
        elif i==']':
            stack[-1].files.sort()
            stack.pop()
    stack[-1].files.sort()
    root=stack.pop()
    return root

def show(root,pre):
    global ans
    ans.append(pre+root.val)
    for dir in root.dirs:
        show(dir,pre+'|     ')
    for file in root.files:
        ans.append(pre+file.val)
    return ans


case=0
s=[]
while True:
    l=input()
    if l=='#':
        break
    if l=='*':
        case+=1
        print(f'DATA SET {case}:')
        root=build(s)
        ans=[]
        show(root,'')
        for i in ans:
            print(i)
        s=[]
        print()
    else:
        s.append(l)