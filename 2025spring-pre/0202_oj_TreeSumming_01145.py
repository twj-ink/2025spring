class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
        self.hasLeft=False

def buildTree(s):
    stack=[]
    node=None
    root=None
    i=0
    while i<len(s):
        if s[i].isdigit():
            curr=s[i]
            i+=1
            while s[i].isdigit():
                curr+=s[i]
                i+=1
            curr=int(curr)

            node=TreeNode(curr)
            if stack:
                if not stack[-1].hasLeft:
                    stack[-1].left=node
                else:
                    stack[-1].right=node
        elif s[i]=='(':
            if node:
                stack.append(node)
                node=None
            i+=1
        elif s[i]==')':
            if stack:
                if not stack[-1].hasLeft:
                    stack[-1].hasLeft=True
                    i+=1
                    node=None
                    continue
                else:
                    root=stack.pop()
            i+=1
        else:
            i+=1
    return root

def check(root,t):
    if not root:
        return False
    if not root.left and not root.right:
        return t==root.val
    t-=root.val
    return check(root.left,t) or check(root.right,t)

finals=[]
paths=[]
path=''
while True:
    try:
        l=input()
        if l[0].isdigit():
            if path:
                paths.append(path)
            path=''
            for i in range(len(l)):
                if l[i]==' ':
                    finals.append(int(l[:i]))
                    path+=l[i+1:]
                    break
        else:
            path+=l.strip()
    except EOFError:
        break
paths.append(path)

for case in range(len(finals)):
    t=finals[case]
    root=buildTree(paths[case])
    if check(root,t):
        print('yes')
    else:
        print('no')