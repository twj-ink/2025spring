class Node:
    def __init__(self,value,next=None):
        self.value=value
        self.next=next

class SinglyLinkedList:
    def __init__(self):
        self.head=None

    def backwardPush(self,value): #append()
        new_node=Node(value)
        if self.head is None:
            self.head=new_node
        else:
            cur=self.head     #通过遍历到达最后一个节点
            while cur.next:   #使用判断cur.next来让cur恰好为最后一个节点
                cur=cur.next  #遍历
            cur.next=new_node

    def insert(self,value,position):
        newNode=Node(value)
        if self.head is None:
            self.head=newNode
        else:
            i=0
            cur=self.head
            while cur.next and i<position-1:
                cur=cur.next
                i+=1
            if cur.next is not None:
                newNode.next=cur.next
                cur.next=newNode
            else:
                cur.next=newNode

    def delete(self,value):
        if self.head is None:
            print(f'Do not find value:{value}')
            return

        if self.head.value==value:
            self.head=self.head.next #直接把head设置成head的next，相当于删除本身
        else:
            matched=False
            cur=self.head
            while cur.next:
                if cur.next.value==value: #把cur放在要删除的节点之前一个位置
                    cur.next.next=cur.next
                    matched=True
                    break
                cur=cur.next
            if not matched:
                print(f'Do not find value: {value}')

    def show(self):
        cur=self.head
        while cur:
            print(cur.value,end='->')
            cur=cur.next
        print('None')


s=SinglyLinkedList()
s.backwardPush(1)
s.backwardPush(2)
s.backwardPush(4)
s.show()
s.delete(3)
s.delete(1)
s.show()
s.insert(2,3)
s.show()

class DoublyLinkedList:
    class Node:
        def __init__(self,value,next=None,prev=None):
            self.value=value
            self.next=None
            self.prev=None

    def __init__(self):
        self.head=None
        self.tail=None

    def append(self,value):
        newNode=Node(value)
        if self.head is None:
            self.head=newNode
            self.tail=newNode
        else:
            self.tail.next=newNode
            newNode.prev=self.tail
            self.tail=newNode

    def prepend(self,value):
        newNode=Node(value)
        if self.head is None:
            self.head=newNode
            self.tail=newNode
        else:
            self.head.prev=newNode
            newNode.next=self.head
            self.head=newNode

    def delete(self,value):
        if not self.head:
            print(f'Do not find value:{value}')
            return
        if self.head.value==value:
            self.head=self.head.next
            if self.head: #如果链表非空
                self.head.prev=None
        elif self.tail.value==value:
            self.tail=self.tail.next
            # if self.tail:
            #     self.tail.next=None
        else:
            cur=self.head
            while cur:
                if cur.value==value:
                    cur.prev.next=cur.next
                    cur.next.prev=cur.prev
                    break
                cur=cur.next

    def show(self):
        print('None',end='<->')
        cur=self.head
        while cur:
            print(cur.value,end='<->')
            cur=cur.next
        print('None')

b=DoublyLinkedList()
b.append(2)
b.append(3)
b.prepend(1)
b.show()
b.delete(2)
b.show()

