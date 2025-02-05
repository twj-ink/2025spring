#pylint:skip-file
class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def buildTree(s):
    stack=[]
    curr=None
    root=None
    is_right_child=False
    for i in s:
        if i=='(': # 说明之前的节点是父节点
            if curr:
                stack.append(curr)
                curr=None
                is_right_child=False
        elif i==')': # 说明stack[-1]处的节点孩子添加完毕
            if curr:
                stack.pop()
        elif i==',':
            is_right_child=True
        elif i.isalpha():
            curr=TreeNode(i)
            if not root:
                root=curr
            if stack:
                if is_right_child:
                    stack[-1].right=curr
                else:
                    stack[-1].left=curr

    return root

def pre(root):
    if root:
        pre_ans.append(root.val)
        pre(root.left)
        pre(root.right)


def inorder(root):
    if root:
        inorder(root.left)
        in_ans.append(root.val)
        inorder(root.right)

for _ in range(int(input())):
    s=input()
    treeRoot=buildTree(s)
    pre_ans,in_ans=[],[]
    pre(treeRoot)
    inorder(treeRoot)
    print(''.join(pre_ans))
    print(''.join(in_ans))