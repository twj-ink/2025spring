class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

n=int(input())
nodes=[TreeNode(i) for i in range(n)]
hasparent=[False]*(n)
isleave=[False]*n

for i in range(n):
    l,r=map(int,input().split())
    if l!=-1:
        nodes[i].left=nodes[l]
        hasparent[l]=True
    if r!=-1:
        nodes[i].right=nodes[r]
        hasparent[r]=True
    if l==-1 and r==-1:
        isleave[i]=True

for i in range(n):
    if hasparent[i]==False:
        root=nodes[i]

def height(root):
    if not root:
        return 0
    return max(height(root.left),height(root.right))+1
H=height(root)-1

leaves=sum(1 for i in range(n) if isleave[i]==True)

print(H,leaves)

