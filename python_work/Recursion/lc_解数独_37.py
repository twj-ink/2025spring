from typing import List
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 记录行、列、3x3宫中已经使用的数字
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]

        # 初始化已经填入的数字
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    val = board[i][j]
                    rows[i].add(val)
                    cols[j].add(val)
                    boxes[(i // 3) * 3 + (j // 3)].add(val)

        # 回溯函数
        def backtrack(i, j):
            if i == 9:  # 如果行遍历完成，说明已解答完毕
                # for p in board: print(*p)
                return True
            if j == 9:  # 如果列遍历完成，跳到下一行
                return backtrack(i + 1, 0)
            if board[i][j] != '.':  # 已经填充过的格子，跳过
                return backtrack(i, j + 1)

            # 尝试填充每个数字 1-9
            for num in range(1, 10):
                num_str = str(num)
                box_idx = (i // 3) * 3 + (j // 3)
                if num_str not in rows[i] and num_str not in cols[j] and num_str not in boxes[box_idx]:
                    # 填充数字
                    board[i][j] = num_str
                    rows[i].add(num_str)
                    cols[j].add(num_str)
                    boxes[box_idx].add(num_str)

                    # 继续回溯
                    if backtrack(i, j + 1):
                        return True

                    # 回溯时撤销选择
                    board[i][j] = '.'
                    rows[i].remove(num_str)
                    cols[j].remove(num_str)
                    boxes[box_idx].remove(num_str)

            return False

        # 从(0,0)开始回溯
        backtrack(0, 0)

if __name__ == '__main__':
    s=Solution()
    print(s.solveSudoku([["5","3",".",".","7",".",".",".","."],
                         ["6",".",".","1","9","5",".",".","."],
                         [".","9","8",".",".",".",".","6","."],
                         ["8",".",".",".","6",".",".",".","3"],
                         ["4",".",".","8",".","3",".",".","1"],
                         ["7",".",".",".","2",".",".",".","6"],
                         [".","6",".",".",".",".","2","8","."],
                         [".",".",".","4","1","9",".",".","5"],
                         [".",".",".",".","8",".",".","7","9"]]))