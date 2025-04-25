class TreeNode:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right
        self.pl=None
        self.pr=None

def find(node):
    if not node or not node.left:
        return node
    return find(node.left)

def solve():
    n,m=map(int,input().split())
    nodes=[TreeNode(i) for i in range(n)]
    for i in range(n):
        idx,l,r=map(int,input().split())
        if l!=-1:
            nodes[idx].left=nodes[l]
            nodes[l].pl=nodes[idx]
        if r!=-1:
            nodes[idx].right=nodes[r]
            nodes[r].pr=nodes[idx]

    for _ in range(m):
        l=list(map(int,input().split()))
        if l[0]==1:
            x,y=nodes[l[1]],nodes[l[2]]
            if x.pl is not None:
                if y.pl is not None:
                    x.pl.left,y.pl.left=y.pl.left,x.pl.left
                    x.pl,y.pl=y.pl,x.pl
                elif y.pr is not None:
                    x.pl.left,y.pr.right=y.pr.right,x.pl.left
                    x.pr,y.pl=y.pr,x.pl
                    x.pl=y.pr=None
            elif x.pr is not None:
                if y.pl is not None:
                    x.pr.right,y.pl.left=y.pl.left,x.pr.right
                    x.pl,y.pr=y.pl,x.pr
                    x.pr=y.pl=None
                elif y.pr is not None:
                    x.pr.right,y.pr.right=y.pr.right,x.pr.right
                    x.pr,y.pr=y.pr,x.pr
        else:
            node=l[1]
            res=find(nodes[node])
            print(res.val)

def main():
    t=int(input())
    for _ in range(t):
        solve()

main()
