from bisect import bisect_left,bisect_right

# s=[1,2,3,4,5]
# print(bisect_left(s,6))
# print(bisect_right(s,3))
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # m=len(matrix[0])
        # for i in matrix:
        #     idx=bisect_left(i,target)
        #     if 0<=idx<m and i[idx]==target:
        #         return True
        # return False

        n,m=len(matrix),len(matrix[0])
        x,y=0,m-1
        while x<n and y>=0:
            if matrix[x][y]==target:
                return True
            elif matrix[x][y]>target:
                y-=1
            else:
                x+=1
        return False