from typing import List
# # from math import gcd
#
# class Fraction:
#     def __init__(self,top,bottom):
#         x=self.gcd(top,bottom)
#         self.top=top//x
#         self.bottom=bottom//x
#
#     # def show(self):
#     #     print(self.top,'/',self.bottom)
#
#     def __str__(self):
#         return str(self.top)+'/'+str(self.bottom)
#
#     def __add__(a,b):
#         new_top=a.top*b.bottom+a.bottom*b.top
#         new_bottom=a.bottom*b.bottom
#         return Fraction(new_top,new_bottom)
#
#     @staticmethod
#     def gcd(a,b):
#         while b!=0:
#             a,b=b,a%b
#         return a
#
#     def __eq__(self,b):
#         return self.top*b.bottom==self.bottom*b.top
#
# a=Fraction(1,4)
# b=Fraction(1,4)
# print(a==b)
###########################################################
# import time
# s=[[1 for i in range(10**6)] for _ in range(10**6)]
# start_time1 = time.process_time()  # 记录起始 CPU 时间
# # 模拟要运行的代码
# k=0
# for i in range(10):
#     for j in range(10):
#         k+=s[i][j]
# print(k)
# end_time1 = time.process_time()  # 记录结束 CPU 时间
#
# print(f"CPU 运行时间: {end_time1 - start_time1:.20f} 秒")
#
# start_time2 = time.process_time()  # 记录起始 CPU 时间
# # 模拟要运行的代码
# k=0
# for j in range(10):
#     for i in range(10):
#         k+=s[i][j]
# print(k)
# end_time2 = time.process_time()  # 记录结束 CPU 时间
#
# print(f"CPU 运行时间: {end_time2 - start_time2:.20f} 秒")
############################################################
# n=int(input())
# a=list(map(int,input().split()))
# b=list(map(int,input().split()))
# dp=[[0]*n for _ in range(n)]
# dp[0][0]=1 if a[0]==b[0] else 0
# for i in range(1,n):
#     dp[i][0]=max(dp[i-1][0]+1,dp[i][0]) if a[i]==b[0] else dp[i-1][0]
# for j in range(1,n):
#     dp[0][j]=max(dp[0][j-1]+1,dp[0][j]) if b[j]==a[0] else dp[0][j-1]
# for i in range(1,n):
#     for j in range(1,n):
#         if a[i]==b[j]:
#             dp[i][j]=max(dp[i-1][j-1]+1,dp[i-1][j],dp[i][j-1])
#         else:
#             dp[i][j]=max(dp[i-1][j],dp[i][j-1])
# print(dp[-1][-1])
##########################################################

### Codeforces Round 998(Div.3) ###
### A ###
# for _ in range(int(input())):
#     a,b,c,d=map(int,input().split())
#     s=set()
#     s.add(a+b)
#     s.add(c-b)
#     s.add(d-c)
#     print(4-len(s))
#

### B ###
# for _ in range(int(input())):
#     n,m=map(int,input().split())
#     s=[]
#     for _ in range(n):
#         s.append((_+1,(sorted(list(map(int,input().split()))))))
#         s.sort(key=lambda x:x[1])
#     curr=0
#     f=True
#     for j in range(m):
#         if f:
#             for i in range(n):
#                 if s[i][1][j]!=curr:
#                     f=False
#                     break
#                 curr+=1
#     ans=[]
#     for i in range(n):
#         ans.append(s[i][0])
#     print(' '.join(map(str,ans)) if f else -1)

### C ###
# for _ in range(int(input())):
#     n,k=map(int,input().split())
#     s=sorted(list(map(int,input().split())))
#     i,j=0,n-1
#     cnt=0
#     while i<j:
#         if s[i]+s[j]<k:
#             i+=1
#         elif s[i]+s[j]>k:
#             j-=1
#         else:
#             i+=1
#             j-=1
#             cnt+=1
#     print(cnt)

### D ###
# for _ in range(int(input())):
#     n=int(input())
#     s=list(map(int,input().split()))
#     if s==sorted(s) or n==1:
#         print('YES')
#         continue
#     if n>=2 and s[0]>s[1]:
#         print('NO')
#         continue
#     for i in range(n-2):
#         s[i+1]-=s[i]
#         if s[i+1]>s[i+2]:
#             print('NO')
#             break
#     else:
#         print('YES')
##################################################
# import math
#
# # 快速幂计算 (a^b) % mod
# def fast_power(a, b, mod):
#     return pow(a,b,mod)
# # 主逻辑
# def solve_mersenne_number(p):
#     # 1. 计算位数
#     log10_2 = math.log10(2)
#     digits = math.floor(p * log10_2) + 1
#
#     # 2. 计算最后 500 位
#     mod = 10**500
#     last_500_digits = (fast_power(2, p, mod) - 1) % mod
#
#     # 3. 格式化最后 500 位
#     last_500_str = f"{last_500_digits:0500d}"  # 补零到 500 位
#
#     # 输出结果
#     print(digits)
#     for i in range(0, 500, 50):
#         print(last_500_str[i:i+50])
#
# # 读取输入并调用主逻辑
# if __name__ == "__main__":
#     p = int(input().strip())
#     solve_mersenne_number(p)
####################################################
# name,weight='Alice',120.2
# print(f'{name} weighs {weight:.2f}')
# n=42
# print(f'{n:x>10}')
# n=255222
# print(f'二进制: {n:b}, 八进制: {n:o}, 十六进制 {n:x}')
# print(f'{n:;}')
###################################################
print(0^0)