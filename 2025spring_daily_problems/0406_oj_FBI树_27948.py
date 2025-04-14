class TreeNode:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None

type={0:'B',1:'I',2:'F'}
def check(s):
    if set(s)=={'0'}: return type[0]
    elif set(s)=={'1'}: return type[1]
    return type[2]
def build(s):
    val=check(s)
    root=TreeNode(val)
    if len(s)>1:
        root.left=build(s[:len(s)//2])
        root.right=build(s[len(s)//2:])
    return root
def pos(root):
    if root: return pos(root.left)+pos(root.right)+root.val
    return ''

n=int(input())
s=list(input())
root=build(s)
print(pos(root))