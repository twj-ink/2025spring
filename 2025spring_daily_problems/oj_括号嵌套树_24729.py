class TreeNode:
    def __init__(self,val,child=None):
        self.val=val
        self.child=child if child is not None else []

def build(s):
    stack=[]
    if not s:
        return None
    if len(s)==1: return TreeNode(s[0])
    # root=TreeNode(s[0])
    curr=None

    for i in s:
        if i=='(':
            stack.append(curr)
        elif i.isalpha():
            curr=TreeNode(i)
            if stack:
                stack[-1].child.append(curr)
        elif i==')':
            root=stack.pop()

    return root

def pre(root,ans):
    if root:
        if isinstance(root,TreeNode):
            ans.append(root.val)
            pre(root.child,ans)
        else:
            for node in root:
                ans.append(node.val)
                pre(node.child,ans)
    return ''

def post(root,ans):
    if root:
        if isinstance(root,TreeNode):
            post(root.child,ans)
            ans.append(root.val)
        else:
            for node in root:
                post(node.child,ans)
                ans.append(node.val)
    return ''


s=input()
root=build(s)
# print(root.child)
ans1,ans2=[],[]
pre(root,ans1)
post(root,ans2)
print(''.join(ans1))
print(''.join(ans2))
