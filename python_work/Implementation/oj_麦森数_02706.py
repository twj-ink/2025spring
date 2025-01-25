import math

# 快速幂计算 (a^b) % mod
def fast_power(a, b, mod):
    return pow(a,b,mod)
# 主逻辑
def solve_mersenne_number(p):
    # 1. 计算位数
    log10_2 = math.log10(2)
    digits = math.floor(p * log10_2) + 1

    # 2. 计算最后 500 位
    mod = 10**500
    last_500_digits = (fast_power(2, p, mod) - 1) % mod

    # 3. 格式化最后 500 位
    last_500_str = f"{last_500_digits:0500d}"  # 补零到 500 位

    # 输出结果
    print(digits)
    for i in range(0, 500, 50):
        print(last_500_str[i:i+50])

# 读取输入并调用主逻辑
if __name__ == "__main__":
    p = int(input().strip())
    solve_mersenne_number(p)

