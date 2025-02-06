# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        def bfs(root):
            nonlocal ans
            q=deque()
            q.append(root)
            while q:
                curr=[]
                for _ in range(len(q)):
                    node=q.popleft()
                    if node.left:
                        q.append(node.left)
                        curr.append(node.left.val)
                    if node.right:
                        q.append(node.right)
                        curr.append(node.right.val)
                if curr:
                    ans.append(curr[:])
        ans=[]
        if not root:
            return ans
        ans.append([root.val])
        bfs(root)
        return ans