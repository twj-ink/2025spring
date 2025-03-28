from collections import deque

class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def buildTree(s):
    if len(s)==0:
        return None
    if len(s)==1:
        return TreeNode(s[0])
    root=TreeNode(s[0])
    def helper(root,i):
        node=TreeNode(i)
        if i<root.val:
            if root.left is None:
                root.left=node
            else:
                helper(root.left,i)
        elif i>root.val:
            if root.right is None:
                root.right=node
            else:
                helper(root.right,i)

    for i in range(1,len(s)):
        helper(root,s[i])
    return root

def level_traversal(root):
    q=deque([root])
    ans=[root.val]
    while q:
        node=q.popleft()
        if node.left:
            q.append(node.left)
            ans.append(node.left.val)
        if node.right:
            q.append(node.right)
            ans.append(node.right.val)
    return ans

s=list(map(int,input().split()))
root=buildTree(s)
print(*level_traversal(root))

def insert(node, value):
    if node is None:
        return TreeNode(value)
    if value < node.value:
        node.left = insert(node.left, value)
    elif value > node.value:
        node.right = insert(node.right, value)
    return node
