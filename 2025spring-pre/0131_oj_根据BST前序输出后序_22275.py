#pylint:skip-file
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
        if i<root.val:
            if root.left is None:
                root.left=TreeNode(i)
            else:
                helper(root.left,i)
        else:
            if root.right is None:
                root.right=TreeNode(i)
            else:
                helper(root.right,i)
    for i in range(1,len(s)):
        helper(root,s[i])
    return root

def postorder(root):
    global ans
    if root:
        postorder(root.left)
        postorder(root.right)
        ans+=[root.val]
    return ans

input()
s=list(map(int,input().split()))
root=buildTree(s)
ans=[]
print(*postorder(root))

#事实上，对数组排序就是中序了

#或者用快速排序选基准，而基准就是第一个元素的根节点
def post_order(pre_order):
    if not pre_order:
        return []
    root = pre_order[0]
    left_subtree = [x for x in pre_order if x < root]
    right_subtree = [x for x in pre_order if x > root]
    return post_order(left_subtree) + post_order(right_subtree) + [root]

n = int(input())
pre_order = list(map(int, input().split()))
print(' '.join(map(str, post_order(pre_order))))