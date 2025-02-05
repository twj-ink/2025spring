# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        # 颜色填充法，新的是白色，放入栈时变为灰色，弹出时若为灰色就输出，白色说明的新的，重复前面步骤
        white, gray = 0, 1
        res = []
        stack = [(white, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == white:
                stack.append((white, node.right))
                stack.append((gray, node))
                stack.append((white, node.left))
            else:
                res.append(node.val)
        return res
        # ans=[]
        # def inorder(root):
        #     if not root:
        #         return
        #     inorder(root.left)
        #     ans.append(root.val)
        #     inorder(root.right)
        # inorder(root)
        # return ans
