def main():
    t = int(input())  # 读取测试用例数量
    for _ in range(t):
        n, m = map(int, input().split())  # 读取 n 和 m
        a = list(map(int, input().split()))  # 读取颜色容量数组 a
        # 构造频次数组，记录每种容量的出现次数
        freq = [0] * (n + 2)  # 大小为 n+2，方便后续计算
        for cap in a:
            freq[cap] += 1
        # 构造后缀和数组 F[x]，表示容量 >= x 的颜色数量
        F = [0] * (n + 2)
        F[n] = freq[n]  # 初始化最后一个位置
        for x in range(n - 1, 0, -1):
            F[x] = F[x + 1] + freq[x]  # 计算后缀和
        # 计算结果
        res = 0
        for i in range(1, n):
            left_needed = i  # 左侧需要容量 >= i
            right_needed = n - i  # 右侧需要容量 >= n-i
            # 计算满足条件的颜色数量
            count = F[left_needed] * F[right_needed] - F[max(left_needed, right_needed)]
            res += count  # 累加到结果中
        print(res)  # 输出结果

if __name__ == '__main__':
    main()
