class Solution:
    def constructProductMatrix(self, s: List[List[int]]) -> List[List[int]]:
        n, m, mod = len(s), len(s[0]), 12345
        ans = [[1] * m for _ in range(n)]
        suf = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                ans[i][j] = suf % mod
                suf = (suf * s[i][j]) % mod

        pre = 1
        for i in range(n):
            for j in range(m):
                ans[i][j] = (ans[i][j] * pre) % mod
                pre = (pre * s[i][j]) % mod

        return ans