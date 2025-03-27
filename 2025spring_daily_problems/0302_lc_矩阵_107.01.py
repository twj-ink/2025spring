import collections


class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        n, m = len(mat), len(mat[0])
        ans = [[0] * m for _ in range(n)]
        s = []
        for i in range(n):
            for j in range(m):
                if mat[i][j] == 0:
                    s.append((i, j))
        step = 1
        dx, dy = [0, -1, 1, 0], [-1, 0, 0, 1]
        inq = set(s)
        s = collections.deque(s)

        while s:
            for _ in range(len(s)):
                x, y = s.popleft()
                for i in range(4):
                    nx, ny = x + dx[i], y + dy[i]
                    if 0 <= nx < n and 0 <= ny < m and mat[nx][ny] == 1 and (nx, ny) not in inq:
                        ans[nx][ny] = step
                        s.append((nx, ny))
                        inq.add((nx, ny))
            step += 1
        return ans