# from collections import deque
# for _ in range(int(input())):
#     n=int(input())
#     s=deque()
#     for _ in range(n):
#         t,x=map(int,input().split())
#         if t==1:
#             s.append(x)
#         else:
#             if not s:
#                 continue
#             elif x==0:
#                 s.popleft()
#             else:
#                 s.pop()
#     print(' '.join(map(str,s)) if s else 'NULL')
class Node:
    def __init__(self,value,next,prev):
        self.value=value
        self.next=next
        self.prev=prev

class Deque:
    def __init__(self):
        self.head=None
        self.tail=None

    def append(self,val):
        nd=Node(val,None,None)
        if not self.head:
            self.head=nd
            self.tail=nd
        else:
            nd.prev=self.tail
            self.tail.next=nd
            self.tail=nd

    def appendleft(self,val):
        nd=Node(val,None,None)
        if not self.head:
            self.head=nd
            self.tail=nd
        else:
            nd.next=self.head
            self.head.prev=nd
            self.head=nd

    def pop(self):
        if not self.head:
            return None
        should_return=self.tail
        new_tail=self.tail.prev
        if self.head==self.tail:
            self.head=self.tail=None
            return should_return
        self.tail.prev.next=None
        self.tail=new_tail
        return should_return

    def popleft(self):
        if not self.head:
            return None
        should_return=self.head
        new_head=self.head.next
        if self.head==self.tail:
            self.head=self.tail=None
            return should_return
        self.head.next.prev=None
        self.head=new_head
        return should_return

    def show(self):
        if self.head==None:
            print('NULL')
        else:
            cur=self.head
            ans=''
            while cur:
                ans+=str(cur.value)+' '
                cur=cur.next
            print(ans.rstrip())

for _ in range(int(input())):
    n=int(input())
    d=Deque()
    for _ in range(n):
        type,num=map(int,input().split())
        if type==1:
            d.append(num)
        else:
            if num==0:
                d.popleft()
            else:
                d.pop()
    d.show()