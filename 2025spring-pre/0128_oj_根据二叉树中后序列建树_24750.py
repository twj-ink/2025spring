# 后序的最后一个是根节点，找到中序这个根节点，其左边右边为左子树和右子树

class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def buildTree(inorder,postorder):
    if len(inorder)==1:
        return TreeNode(inorder[0])
    if len(inorder)==0 or len(postorder)==0:
        return None
    root=TreeNode(postorder[-1])
    rootIdx=inorder.index(root.val)

    root.left=buildTree(inorder[:rootIdx],postorder[:rootIdx])
    root.right=buildTree(inorder[rootIdx+1:],postorder[rootIdx:-1])

    return root

def preorderTraversal(root):
    if root:
        return root.val+preorderTraversal(root.left)+preorderTraversal(root.right)
    return ''

inorder=input()
postorder=input()
root=buildTree(inorder,postorder)
print(preorderTraversal(root))