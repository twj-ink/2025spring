class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def buildTree(s):
    def helper():
        nonlocal i
        if i>=len(s):
            return None
        char=s[i]
        i+=1
        if char=='.':
            return None
        node=TreeNode(char)
        node.left=helper()
        node.right=helper()
        return node

    i=0
    return helper()

def inorder(root):
    if root:
        return inorder(root.left)+root.val+inorder(root.right)
    return ''


def postorder(root):
    if root:
        return postorder(root.left)+postorder(root.right)+root.val
    return ''


s=input()
root=buildTree(s)
print(inorder(root))
print(postorder(root))