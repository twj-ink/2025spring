class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def build(preorder,inorder):
    if len(preorder)==0:
        return None
    if len(preorder)==1:
        return TreeNode(preorder[0])

    root=TreeNode(preorder[0])
    idx=inorder.find(root.val)
    root.left=build(preorder[1:idx+1],inorder[:idx])
    root.right=build(preorder[idx+1:],inorder[idx+1:])
    return root

def postorder(root):
    if root:
        return postorder(root.left)+postorder(root.right)+root.val
    return ''

while True:
    try:
        a,b=input().split()
        root=build(a,b)
        print(postorder(root))

    except EOFError:
        break