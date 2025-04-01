# Definition for a binary tree node.
from typing import Optional
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(root,left,right):
            if not root:
                return True
            val = root.val
            return left<val<right and dfs(root.left,left,val) and dfs(root.right,val,right)
        return dfs(root,-float('inf'),float('inf'))




        # prev=-float('inf')
        # def inorder(root):
        #     nonlocal prev
        #     if root:
        #         if not inorder(root.left):
        #             return False
        #         if root.val<=prev:
        #             return False
        #         prev=root.val
        #         if not inorder(root.right):
        #             return False
        #
        #     return True
        # return inorder(root)


