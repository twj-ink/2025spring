from collections import deque
class TreeNode:
    def __init__(self,val=None,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def getTree(s):
    stack=[]
    for i in s:
        if ord('a')<=ord(i)<=ord('z'):
            stack.append(TreeNode(i))
        else:
            r=stack.pop()
            l=stack.pop()
            node=TreeNode(i)
            node.left=l
            node.right=r
            stack.append(node)
    return stack[0]

def bfs(root):
    q=deque()
    q.append(root)
    ans=deque()
    ans.appendleft(root.val)
    while q:
        node=q.popleft()
        if node.left:
            q.append(node.left)
            ans.appendleft(node.left.val)
        if node.right:
            q.append(node.right)
            ans.appendleft(node.right.val)
    return ''.join(ans)

n=int(input())
for _ in range(n):
    s=input()
    root=getTree(s)
    print(bfs(root))