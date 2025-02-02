# a,b=map(int,input().split())
# s=[input() for _ in range(a)]
# t=[input() for _ in range(b)]
# m=t[0]
# f=True
# for i in range(a-b+1):
#     if f:
#         curr=i
#         for j in range(a-b+1):
#             if m==s[curr][j:j+b] and all(t[k]==s[curr+k][j:j+b] for k in range(1,b)):
#                 print(i+1,j+1)
#                 f=False
#                 break
MOD = 998244353


# 计算组合数 C(n, k) % MOD
def comb(n, k, fact, inv_fact):
    if k > n or k < 0:
        return 0
    return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD


# 计算阶乘和阶乘逆元
def precompute_factorials(limit):
    fact = [1] * (limit + 1)
    inv_fact = [1] * (limit + 1)
    for i in range(2, limit + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact[limit] = pow(fact[limit], MOD - 2, MOD)
    for i in range(limit - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    return fact, inv_fact


def solve(N, M, S):
    fact, inv_fact = precompute_factorials(max(N, M))

    result = [0] * (N + 1)

    # 对每个 k = 0, 1, ..., N 计算答案
    for k in range(N + 1):
        # 选择 k 个字符与 S 匹配
        num_ways_to_choose_k = comb(N, k, fact, inv_fact)

        # 剩下的 M - k 个字符，必须是 S 中没有的字符
        num_ways_to_fill_remaining = pow(26 - 1, M - k, MOD)

        # 最终结果
        result[k] = num_ways_to_choose_k * num_ways_to_fill_remaining % MOD

    return result


def main():
    N, M = map(int, input().split())
    S = input().strip()

    result = solve(N, M, S)

    print(" ".join(map(str, result)))


if __name__ == "__main__":
    main()
