class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def height(root):
    if not root:
        return 0
    return max(height(root.left),height(root.right))+1

n=int(input())
tree=[TreeNode(i) for i in range(n+1)]
for i in range(n):
    l,r=map(int,input().split())
    if l!=-1:
        tree[i+1].left=tree[l]
    if r!=-1:
        tree[i+1].right=tree[r]
root=tree[1]
# print(root)
h=height(root)
print(h)