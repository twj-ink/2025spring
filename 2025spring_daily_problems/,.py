for _ in range(int(input())):
    n = int(input())  # 读入点的数量
    ps = set()  # 存储所有点
    for _ in range(n):
        x, y = map(int, input().split())  # 读入每个点
        ps.add((x, y))

    cnt = 0
    # 转换为列表来遍历所有的点对
    s = list(ps)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            x1, y1 = s[i]
            x2, y2 = s[j]

            # 只考虑x坐标和y坐标不同的点对
            if x1 != x2 and y1 != y2:
                # 检查另外两个点是否在集合中
                if (x1, y2) in ps and (x2, y1) in ps:
                    cnt += 1

    print(cnt//2)
