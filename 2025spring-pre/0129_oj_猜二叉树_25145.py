from collections import deque

class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def build_tree(inorder,postorder):
    if len(inorder)==1:
        return TreeNode(inorder[0])
    if not inorder:
        return None

    root=TreeNode(postorder[-1])
    idx=inorder.find(root.val)
    root.left=build_tree(inorder[:idx],postorder[:idx])
    root.right=build_tree(inorder[idx+1:],postorder[idx:-1])
    return root

def level_traversal(root):
    q=deque([root])
    ans=[root.val]
    while q:
        root=q.popleft()
        if root.left:
            q.append(root.left)
            ans.append(root.left.val)
        if root.right:
            q.append(root.right)
            ans.append(root.right.val)
    return ''.join(ans)

n=int(input())
for _ in range(n):
    inorder=input()
    postorder=input()
    root=build_tree(inorder,postorder)
    print(level_traversal(root))