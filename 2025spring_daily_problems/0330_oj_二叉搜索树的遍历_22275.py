from collections import deque

class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val = val
        self.left=left
        self.right=right

def insert_node(root,val):
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_node(root.left,val)
    elif val > root.val:
        root.right = insert_node(root.right,val)
    return root

def post(root):
    if root:
        return post(root.left)+post(root.right)+[root.val]
    return []

n=int(input())
s=list(map(int,input().split()))
root = None
for i in s:
    root = insert_node(root,i)
ans=post(root)
print(*ans)
