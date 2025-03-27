from typing import List
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        n=numRows
        ans=[[1],[1,1]]
        if n==1:
            return [[1]]
        if n==2:
            return [[1],[1,1]]
        for i in range(2,n):
            path=[1]+[0]*(i-1)+[1]
            for j in range(1,i):
                path[j]=ans[i-1][j-1]+ans[i-1][j]
            ans.append(path)
        return ans
if __name__ == '__main__':
    s=Solution()
    print(s.generate(5))