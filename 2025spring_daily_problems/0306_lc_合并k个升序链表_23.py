# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # def merge(l1,l2):
        #     dummy = ListNode(0)
        #     curr = dummy
        #     while l1 and l2:
        #         if l1.val <= l2.val:
        #             curr.next, l1 = l1, l1.next
        #         else:
        #             curr.next, l2 = l2, l2.next
        #         curr = curr.next
        #     curr.next = l1 if l1 else l2
        #     return dummy.next

        # def mergeSort(lists):
        #     n=len(lists)
        #     if n==1:
        #         return lists[0]
        #     if n==0:
        #         return None
        #     mid=n//2
        #     left=mergeSort(lists[:mid])
        #     right=mergeSort(lists[mid:])
        #     return merge(left,right)

        # return mergeSort(lists)

        def __lt__(self, other):
            return self.val < other.val

        ListNode.__lt__ = __lt__
        heap = []
        for i, l in enumerate(lists):
            if l:
                heappush(heap, (l.val, i, l))
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, i, l = heappop(heap)
            curr.next = ListNode(val)
            curr = curr.next
            if l.next:
                l = l.next
                heappush(heap, (l.val, i + 1, l))
        return dummy.next