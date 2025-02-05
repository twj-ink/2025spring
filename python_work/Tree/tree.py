class TreeNode:
    def __init__(self,val=0,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
# root=TreeNode()
# 前序遍历
def preorder_traversal(root):
    if root:
        print(root.val)
        preorder_traversal(root.left)
        preorder_traversal(root.right)

# 中序遍历
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.val)
        inorder_traversal(root.right)

# 后序遍历
def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)
        postorder_traversal(root.right)
        print(root.val)

# 顺层级遍历(BFS)--从上到下，从左到右
from collections import deque
def level_order_traversal(root):
    if not root:
        return []
    q=deque()
    q.append(root)
    res=[]

    while q:
        node=q.popleft()
        res.append(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res

# 逆层级遍历(BFS)--从下到上，从右到左
def reverse_level_order_traversal(root):
    if not root:
        return []
    q=deque()
    q.append(root)
    res=deque() # 方便从左侧进入元素

    while q:
        node=q.popleft()
        res.appendleft(node.val)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return list(res)

# 求树的高度(深度)
def tree_height(root):
    if not root:
        return 0
    left_height=tree_height(root.left)
    right_height=tree_height(root.right)
    return max(left_height,right_height)+1

# 判断两棵树是否相同
def is_same(p,q):
    if not p and not q:
        return True
    if not p or not q:
        return False
    return (p.val==q.val and
            is_same(p.left,q.left) and
            is_same(p.right,q.right))

# 反转二叉树
def invert_tree(root):
    if root:
        root.left,root.right=invert_tree(root.right),invert_tree(root.left)
    return root

# 对于二叉搜索树BST，寻找最小值和最大值，(这个递归过程与链表的寻找最后一个节点是类似的)
# 最小值在最左边
def find_min(root):
    if not root.left:
        return root.val
    return find_min(root.left)

# 最大值在最右边
def find_max(root):
    if not root.right:
        return root.val
    return find_max(root.right)

# 判断是否为平衡二叉树
# 对于所有节点，其左子树和右子树的高度差不超过1
def is_balanced(root):
    def check_height(root):
        if not root:
            return 0
        left_height=check_height(root.left)
        if left_height==-1:
            return -1
        right_height=check_height(root.right)
        if right_height==-1 or abs(left_height-right_height)>1:
            return -1
        return max(left_height,right_height)+1

    return check_height(root) != -1


