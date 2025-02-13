# 前序的第一个是根节点，找到中序这个根节点，其左边右边为左子树和右子树

class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def buildTree(preorder,inorder):
    if len(inorder)==1:
        return TreeNode(inorder[0])
    if len(inorder)==0 or len(preorder)==0:
        return None
    root=TreeNode(preorder[0])
    rootIdx=inorder.index(root.val)

    root.left=buildTree(preorder[1:rootIdx+1],inorder[:rootIdx])
    root.right=buildTree(preorder[rootIdx+1:],inorder[rootIdx+1:])

    return root

def postorderTraversal(root):
    if root:
        return postorderTraversal(root.left)+postorderTraversal(root.right)+root.val
    return ''

while True:
    try:
        preorder=input()
        inorder=input()
        root=buildTree(preorder,inorder)
        print(postorderTraversal(root))
    except EOFError:
        break