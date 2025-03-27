class Solution:
    def orangesRotting(self, s: List[List[int]]) -> int:
        dx, dy = [0, -1, 1, 0], [-1, 0, 0, 1]
        n, m = len(s), len(s[0])
        cnt = 0
        q = deque()
        inq = set()
        for i in range(n):
            for j in range(m):
                if s[i][j] == 1:
                    cnt += 1
                elif s[i][j] == 2:
                    q.append((i, j))
                    inq.add((i, j))
        if not q and not cnt:
            return 0

        time = 0
        while q:
            for _ in range(len(q)):
                x, y = q.popleft()
                for i in range(4):
                    nx, ny = x + dx[i], y + dy[i]
                    if 0 <= nx < n and 0 <= ny < m and s[nx][ny] == 1 and (nx, ny) not in inq:
                        q.append((nx, ny))
                        inq.add((nx, ny))
                        cnt -= 1
            time += 1

        if cnt == 0:
            return time - 1
        return -1
