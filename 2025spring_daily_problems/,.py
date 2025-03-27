class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
        self.hasleft=False

def build(s):
    stack=[]
    if not s:
        return None
    if len(s)==1: return TreeNode(s[0])
    # root=TreeNode(s[0])
    curr=None

    for i in s:
        if i=='(':
            stack.append(curr)
        elif i.isalpha():
            curr=TreeNode(i)
            if stack:
                if not stack[-1].hasleft:
                    stack[-1].left=curr
                    stack[-1].hasleft=True
                else:
                    stack[-1].right=curr
        elif i=='*':
            stack[-1].hasleft=True
        elif i==')':
            root=stack.pop()

    return root

def pre(root):
    if root:
        return root.val+pre(root.left)+pre(root.right)
    return ''

def ino(root):
    if root:
        return ino(root.left)+root.val+ino(root.right)
    return ''

for _ in range(int(input())):
    s=input()
    root=build(s)
    # print(root.child)
    print(pre(root))
    print(ino(root))
