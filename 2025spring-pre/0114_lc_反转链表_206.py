# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

#迭代法
class Solution:
    def reverseList1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        new_nxt=None
        cur=head
        while cur:
            old_nxt=cur.next
            cur.next=new_nxt
            new_nxt=cur
            cur=old_nxt
        return new_nxt
#递归法
    def reverseList2(self, head: Optional[ListNode]) -> Optional[ListNode]:
        def reverse(head):
            if not head or not head.next:
                return head

            reversed_head=reverse(head.next)
            head.next.next=head
            head.next=None
            return reversed_head
        return reverse(head)