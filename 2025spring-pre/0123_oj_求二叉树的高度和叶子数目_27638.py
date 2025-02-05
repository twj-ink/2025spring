class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def height(root):
    if not root:
        return -1
    left_height=height(root.left)
    right_height=height(root.right)
    return max(left_height,right_height)+1

def count_child(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    return count_child(root.left)+count_child(root.right)

n=int(input())
tree=[TreeNode() for _ in range(n)]
has_parent=[False]*n #无父节点的就是根root
for i in range(n):
    l,r=map(int,input().split())
    if l!=-1:
        tree[i].left=tree[l]
        has_parent[l]=True
    if r!=-1:
        tree[i].right=tree[r]
        has_parent[r]=True
root_idx=has_parent.index(False)
root=tree[root_idx]

treeHeight=height(root)
cnt=count_child(root)

print(treeHeight,cnt)