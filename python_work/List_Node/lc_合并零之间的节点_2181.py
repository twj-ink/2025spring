# 给你一个链表的头节点 head ，该链表包含由 0 分隔开的一连串整数。链表的 开端 和 末尾 的节点都满足 Node.val == 0 。
# 对于每两个相邻的 0 ，请你将它们之间的所有节点合并成一个节点，其值是所有已合并节点的值之和。然后将所有 0 移除，修改后的链表不应该含有任何 0 。
#  返回修改后链表的头节点 head 。

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

from typing import Optional

class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head.next
        while cur:
            if cur.next.val == 0:
                cur.next = cur.next.next
                cur = cur.next
            else:
                cur.val += cur.next.val
                cur.next = cur.next.next
        return head.next

# 合并节点，使用cur作为head.next的引用也是可以的
# 如果可以合并，对cur原地合并并删除下一个，即
# cur.val+=cur.next.val
# cur.next=cur.next.next
# 如果要进入下一个节点，只需删除下一个节点，然后对cur更新
# cur.next=cur.next.next
# cur=cur.next