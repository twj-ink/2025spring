def solve():
    mod = 10 ** 9 + 7
    t = int(input())
    for _ in range(t):
        n = int(input())
        s = input().strip()
        # 当 x = 1 时，不需要任何操作
        if s == "1":
            print(0)
            continue

        # x 可以表示为 2^(n-1) + r，其中 r 为二进制串（去掉最高位）
        r = int(s[1:], 2) if n > 1 else 0

        # 计算分母 2^(n-1) 模 mod 及其逆元
        denom = pow(2, n - 1, mod)
        inv_denom = pow(denom, mod - 2, mod)

        # 预期操作数为 (n - 1) + r/2^(n-1)
        expected = ((n - 1) % mod + (r % mod) * inv_denom) % mod
        print(expected)


if __name__ == '__main__':
    solve()
