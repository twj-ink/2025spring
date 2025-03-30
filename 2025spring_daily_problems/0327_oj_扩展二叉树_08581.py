class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left=left
        self.right=right

def build(s):
    def helper():
        nonlocal i
        if i>=len(s):
            return None
        if s[i]=='.':
            i+=1
            return None

        root=TreeNode(s[i])
        i+=1
        root.left=helper()
        root.right=helper()
        return root
        

    i=0
    root = helper()
    return root

def ino(root):
    if root:
        return ino(root.left)+root.val+ino(root.right)
    return ''

def post(root):
    if root:
        return post(root.left)+post(root.right)+root.val
    return ''

s=input()
root = build(s)
print(ino(root))
print(post(root))
