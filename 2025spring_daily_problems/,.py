# from collections import defaultdict
#
# maxd,d=defaultdict(int),defaultdict(int)
# s=input()
# maxv=0
# prev=s[0]
# d[prev]+=1
# for i in s[1:]:
#     if i==prev:
#         d[i]+=1
#     else:
#         maxd[prev]=max(maxd[prev],d[prev])
#         maxv=max(maxv,maxd[prev])
#         d[prev]=0
#         prev=i
#         d[i]+=1
# maxd[s[-1]]=max(maxd[s[-1]],d[s[-1]])
# maxv=max(maxv,maxd[s[-1]])
#
#
# ans=[]
# for k,v in maxd.items():
#     if v==maxv:
#         ans.append((k,s.find(k)))
# ans.sort(key=lambda x:x[1])
#
# print(ans[0][0],maxd[ans[0][0]])


# d={}
# for i in range(10): d[str(i)]=i
# for i in range(ord('A'),ord('Z')+1): d[chr(i)]=10+i-ord('A')
# dd={v:k for k,v in d.items()}
# def trans(p,n,q):
#     if p!=10: # 转为10进制
#         num=0
#         for i in n:
#             num=num*p+d[i]
#         n=num
#     # print(n)
#     # 转为q
#     ans=''
#     while n!=0:
#         ans+=dd[n%q]
#         n//=q
#     return ans
#
# m=int(input())
# for _ in range(m):
#     p,n,q=input().split(',')
#     p,q=int(p),int(q)
#     print(trans(p,n,q)[::-1])

# print(bin(int('0x00359141',16)))
# print(bin(int('0x4a564504',16)))



# dp[i][j][k]:分配到第i个孩子时，最后一包糖的编号是j，最后一个被复制了的糖的编号是k 时的最大值 的最小可能值

# def main():
#     # 读取 n 和 m
#     n, m = map(int, input().split())
#     # 读取数组 w
#     w = list(map(int, input().split()))
#
#     # 计算前缀和
#     # prefix = [0] * (n + 1)
#     # for i in range(1, n + 1):
#     #     prefix[i] = prefix[i - 1] + w[i - 1]
#
#     w = [0] + w
#     for i in range(1, n + 1):
#         w[i] = w[i] + w[i - 1]
#
#     total = w[n]
#     base = (total * 2) // m  # minn 的上界
#
#     ans = float('inf')
#
#     for minn in range(1, base + 1):
#         # 初始化 DP 表
#         dp = [[[float('inf')] * (n + 1) for _ in range(n + 1)] for _ in range(m + 1)]
#         dp[0][0][0] = minn
#
#         for i in range(1, m + 1):
#             for k in range(n + 1):
#                 p = 0
#                 for j in range(k, n + 1):
#                     if k > 0:
#                         dp[i][j][k] = min(dp[i][j][k], dp[i][j][k - 1])
#                     while p <= j and w[j] - w[p] >= minn:
#                         p += 1
#                     pp = min(p - 1, k)
#                     if pp >= 0:
#                         dp[i][j][k] = min(dp[i][j][k], max(dp[i - 1][k][pp], w[j] - w[pp]))
#
#         ans = min(ans, dp[m][n][n] - minn)
#
#     print(ans)
#
#
# if __name__ == "__main__":
#     main()

# def compute_dp(n, m, prefix, minn):
#     """
#     计算给定 minn 下的 dp 数组，并返回最终结果
#     """
#     dp_prev = [[float('inf')] * (n + 1) for _ in range(n + 1)]
#     dp_prev[0][0] = minn
#
#     for i in range(1, m + 1):
#         dp_curr = [[float('inf')] * (n + 1) for _ in range(n + 1)]
#         for k in range(n + 1):
#             p = 0
#             for j in range(k, n + 1):
#                 if k > 0:
#                     dp_curr[j][k] = min(dp_curr[j][k], dp_curr[j][k - 1])
#                 while p <= j and prefix[j] - prefix[p] >= minn:
#                     p += 1
#                 pp = min(p - 1, k)
#                 if pp >= 0:
#                     dp_curr[j][k] = min(dp_curr[j][k], max(dp_prev[k][pp], prefix[j] - prefix[pp]))
#         dp_prev = dp_curr
#
#     return dp_prev[n][n] if dp_prev[n][n] != float('inf') else None
#
#
# def main():
#     input_data = sys.stdin.read().split()
#     n = int(input_data[0])
#     m = int(input_data[1])
#     w = list(map(int, input_data[2:2 + n]))
#
#     # 计算前缀和
#     prefix = [0] * (n + 1)
#     for i in range(1, n + 1):
#         prefix[i] = prefix[i - 1] + w[i - 1]
#
#     total = prefix[n]
#     min_minn = max(1, total // m - 100)
#     max_minn = min(total, (total * 2) // m + 100)
#
#     ans = float('inf')
#
#     for minn in range(min_minn, max_minn + 1):
#         result = compute_dp(n, m, prefix, minn)
#         if result is not None:
#             ans = min(ans, result - minn)
#
#     print(ans)
#
#
# if __name__ == "__main__":
#     main()


for i in range(1,10):
    for j in range(1,10):
        if i>j:
            print('\t',end='\t')
        else:
            print(f'{i}×{j}={i*j}',end='\t')
    print()

