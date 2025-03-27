# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.ans=0
        def height(root):
            if not root:
                return 0
            L=height(root.left)
            R=height(root.right)
            self.ans=max(self.ans,L+R)
            return max(L,R)+1

        height(root)
        return self.ans