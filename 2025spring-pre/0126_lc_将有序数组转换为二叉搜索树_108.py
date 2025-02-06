# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def helper(l,r):
            if l>r:
                return None
            mid=(l+r)//2
            root=TreeNode(nums[mid])
            root.left=helper(l,mid-1)
            root.right=helper(mid+1,r)
            return root
        return helper(0,len(nums)-1)


# 由于原数组是升序的---就是BST的中序遍历结果
# 所以本题就是要将这个遍历结果转化为BST而已，只不过需要平衡
# 则从中间节点开始，定义的函数返回节点，
# 把该位置节点找好，然后左右节点都用一次递归
# 与 0125-oj-扩展二叉树-08581 类似的递归 ，函数的作用是返回节点
# 则只需要找到跟节点设置一下，然后对子节点递归

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