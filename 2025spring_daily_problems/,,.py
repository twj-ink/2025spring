fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

def dfs(i, total):
    if total == 100:
        return 1
    if total > 100 or i == len(fib):
        return 0
    # 选或不选
    return dfs(i+1, total) + dfs(i+1, total + fib[i])

print(dfs(0, 0))  # 输出为 9
