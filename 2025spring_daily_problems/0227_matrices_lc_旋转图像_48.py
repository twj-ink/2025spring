class Solution:
    def rotate(self, s: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(s)
        for i in range(n):
            for j in range(i):
                s[i][j], s[j][i] = s[j][i], s[i][j]

        half = n // 2
        for j in range(half):
            for i in range(n):
                s[i][j], s[i][n - j - 1] = s[i][n - j - 1], s[i][j]

