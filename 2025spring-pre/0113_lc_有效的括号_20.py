#这个题目的意思是：'[{]}'是错误的，只需要一个栈
from typing import List
class Solution:
    def isValid(self, s: str) -> bool:
        a=[]
        d={')':'(',']':'[','}':'{'}
        for i in s:
            if i in '([{':
                a.append(i)
            else:
                if not a or not a[-1]==d[i]:
                    return False
                a.pop()
        if a:
            return False
        return True

if __name__ == '__main__':
    sol=Solution()
    print(sol.isValid('{}[(]'))