# class TreeNode:
#     def __init__(self,val=None,child=None):
#         self.val=val
#         self.child=child if child else []
#
# def buildTree(s):
#     stack=[]
#     node=None
#     root=None
#     for i in s:
#         if i.isalpha():
#             node=TreeNode(i)
#             if stack:
#                 stack[-1].child.append(node)
#         elif i=='(':
#             stack.append(node)
#             node=None
#         elif i==')':
#             root=stack.pop()
#     return root if root else node
#
# def Preorder(root,ans):
#     if root:
#         ans+=[root.val]
#     if root.child:
#         for c in root.child:
#             Preorder(c,ans)
#     return ans
#
# def Postorder(root,ans):
#     if root.child:
#         for c in root.child:
#             Postorder(c,ans)
#     if root:
#         ans+=[root.val]
#     return ans
#
# s=input()
# root=buildTree(s)
# ans1,ans2=[],[]
# Preorder(root,ans1)
# Postorder(root,ans2)
# print(''.join(ans1))
# print(''.join(ans2))
s=[[1,2,5,0],[3,4,0,4],[4,1,0,0],[4,2,4,1]]
for i in s:
    print(*i)