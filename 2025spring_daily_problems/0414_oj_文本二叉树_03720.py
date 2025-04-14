class TreeNode:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None

def build(s):
    def helper(prefix):
        nonlocal i
        if i<len(s):
            curr=s[i]
            if curr[:-1]==prefix:
                root=TreeNode(curr[-1]) if curr[-1]!='*' else None
                i+=1
                if root:
                    root.left=helper(prefix+'-')
                    root.right=helper(prefix+'-')
                return root
            else:
                return
    i=0
    root=helper('')
    return root


def show(root):
    def pre(root):
        if root:
            return root.val+pre(root.left)+pre(root.right)
        return ''
    def ino(root):
        if root:
            return ino(root.left)+root.val+ino(root.right)
        return ''
    def pos(root):
        if root:
            return pos(root.left)+pos(root.right)+root.val
        return ''
    print(pre(root))
    print(pos(root))
    print(ino(root))


for _ in range(int(input())):
    s=[]
    while (l:=input()) != '0':
        s.append(l)
    root=build(s)
    show(root)
    print()
