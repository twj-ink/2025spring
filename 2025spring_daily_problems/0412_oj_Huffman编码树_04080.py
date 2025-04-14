'''

        0
    0      5
 0    3
1 1
'''

from heapq import heappop,heappush,heapify

class Node:
    def __init__(self,fre,val):
        self.fre=fre
        self.val=val
        self.left=None
        self.right=None

    def __lt__(self, other):
        return self.fre<other.fre

def build(s):
    h=[Node(s[i],s[i]) for i in range(len(s))] #(fre,val)
    heapify(h)

    while len(h)>1:
        left=heappop(h)
        right=heappop(h)
        merged=Node(left.fre+right.fre,None)
        merged.left=left
        merged.right=right
        heappush(h,merged)

    return h[0]

def cal(root,height):
    global ans
    if root.val is None:
        cal(root.left,height+1)
        cal(root.right,height+1)
        return
    ans+=root.val*height
    return

n=int(input())
s=list(map(int,input().split()))
ans=0
root=build(s)
cal(root,0)
print(ans)