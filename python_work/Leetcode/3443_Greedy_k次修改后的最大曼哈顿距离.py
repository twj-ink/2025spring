#先按原始路程计算基数，然后每改变一次方向就+2
#k尽量改变在相反方向的较少数值上

#对每一步逐步计算，不要使用Counter，因为求“路程上的最远距离”
from collections import Counter
class Solution:
    def maxDistance(self,s,k):
        # d=Counter(s)
        # ce,cw,cn,cs=d['E'],d['W'],d['N'],d['S']
        # 初始化计数器和答案
        ce = cw = cn = cs = ans = 0
        for i, ch in enumerate(s):
            if ch == "N": cn += 1
            elif ch == "S": cs += 1
            elif ch == "E": ce += 1
            else: cw += 1

            #当前的基数
            dx=abs(ce-cw)
            dy=abs(cs-cn)
            base=dx+dy

            px=min(ce,cw)
            py=min(cs,cn)
            additional=min(k,px+py)*2
