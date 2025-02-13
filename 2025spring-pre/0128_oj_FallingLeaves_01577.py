# 又是要建树，但是BST满足：
# 左子树<根节点<右子树
# 而这恰好可以使用递归完成

# 使用build_tree来找到第一个根节点，然后对每一个叶子调用insert_node来插入叶子
class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def build_tree(leaves):
    if not leaves:
        return None
    leaves=leaves[::-1]
    # print(leaves)
    root=TreeNode(leaves[0])
    for leaf in leaves[1:]:
        for l in leaf:
            insert_leaf(root,l)
    return root

def insert_leaf(root,leaf):
    if leaf<root.val:
        if root.left: # 如果有左子树了继续向下插入
            insert_leaf(root.left,leaf)
        else: # 如果没有就直接在这一处插入
            root.left=TreeNode(leaf)

    else:
        if root.right:
            insert_leaf(root.right,leaf)
        else:
            root.right=TreeNode(leaf)

def preorder_traversal(root):
    if root:
        return root.val+preorder_traversal(root.left)+preorder_traversal(root.right)
    return ''

all_leaves=[]
leaves=[]
while True:
    l=input()
    if l=='$':
        break
    if l=='*':
        all_leaves.append(leaves)
        leaves=[]
    else:
        leaves.append(l)
all_leaves.append(leaves)
# print(all_leaves)
for leaves in all_leaves:
    root=build_tree(leaves)
    print(preorder_traversal(root))