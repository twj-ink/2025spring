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



s=list(map(int,input().split()))
root = None
for i in s:
    root = insert_node(root,i)

q=deque([root])
#q.append(root.val)
ans=[]
#ans.append(root.val)
while q:
    for _ in range(len(q)):
        node=q.popleft()
        ans.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)

print(*ans)
