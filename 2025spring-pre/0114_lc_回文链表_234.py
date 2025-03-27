# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        stack=[]
        cur=head
        while cur:
            stack.append(cur.val)
            cur=cur.next
        while stack:
            x,y=stack.pop(),head.val
            if x!=y:
                return False
            head=head.next
        return True



        # if not head or not head.next:
        #     return True
        # #find the mid
        # slow=fast=head
        # while fast and fast.next:
        #     slow=slow.next
        #     fast=fast.next.next
        # #reverse the right part
        # new_nxt=None
        # cur=slow
        # while cur:
        #     old_nxt=cur.next
        #     cur.next=new_nxt
        #     new_nxt=cur
        #     cur=old_nxt
        # #new_nxt is the reversed_head
        # #make comparison
        # while new_nxt:
        #     if head.val!=new_nxt.val:
        #         return False
        #     head=head.next
        #     new_nxt=new_nxt.next
        # return True