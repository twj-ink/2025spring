from typing import List

from sympy import factorial

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
# import math
# n=int(input())
# a=math.floor(n**0.5)
# ans=float('inf')
# for i in range(1,a+1):
#     if n/i==n//i:
#         ans=min(ans,2*(i+n//i))
# print(ans)

# print(ord('A'),ord('Z')) 65 90
# s=input()
# ans=''
# for i in s:
#     if ord('A')<=ord(i)<=ord('Z'):
#         p=ord(i)-5
#         if p>=65:
#             ans+=chr(p)
#         else:
#             # print(i,i)
#             r=65-p-1 #e-69,69-5=64,
#             rr=90-r
#             ans+=chr(rr)
#     else:
#         ans+=i
# print(ans)
#
# from decimal import Decimal
# a,b=map(float,input().split())
# ans=a*b/666.667
#
# print(format(ans,'.4f'))

# n=int(input())
# s=list(map(int,input().split()))
# s.reverse()
# while len(s)>1:
#     a=s.pop()
#     b=s.pop()
#     r=abs(a-b)
#     s.append(r)
# print(s[0])

# n=int(input())
# s=[]
# for _ in range(n):
#     a,b=input().split()
#     b=int(b)
#     s.append((a,b))
# s.sort(key=lambda x:x[1])
# for i in s:
#     print(*i)
# dx,dy=[0,-1,1,0],[-1,0,0,1]
# from collections import deque
# def bfs(s,sx,sy,ex,ey,m):
#     q=deque()
#     q.append((sx,sy))
#     inq=set()
#     inq.add((sx,sy))
#     step=0
#     while q:
#         for _ in range(len(q)):
#             x,y=q.popleft()
#             for i in range(4):
#                 nx,ny=x+dx[i],y+dy[i]
#                 if (nx,ny)==(ex,ey):
#                     if step<m:
#                         return 'YES'
#                     else:
#                         return 'NO'
#                 if 0<=nx<n and 0<=ny<n and (nx,ny) not in inq and s[nx][ny]=='.':
#                     inq.add((nx,ny))
#                     q.append((nx,ny))
#             step+=1
#     return 'NO'
#
# for _ in range(int(input())):
#     n,m=map(int,input().split())
#     s=[]
#     for i in range(n):
#         l=input()
#         if 'S' in l:
#             sx=i
#             sy=l.index('S')
#         if 'E' in l:
#             ex=i
#             ey=l.index('E')
#         s.append(l)
#     print(bfs(s,sx,sy,ex,ey,m))

# for _ in range(int(input())):
#     n=int(input())
#     s=[]
#     for _ in range(n):
#         a,b=map(int,input().split())
#         s.append((a,b))
#     s=list(set(s))
#     s.sort()
#     cnt=0
#     if len(s)>=4:
#         for i in range(len(s)-4+1):
#             for j in range(i+1,len(s)-4+2):
#                 if s[i][0]==s[j][0]:
#                     for k in range(j+1,len(s)-4+3):
#                         for p in range(k+1,len(s)):
#                             if s[k][0]==s[p][0] and s[i][1]==s[k][1] and s[j][1]==s[p][1]:
#                                 cnt+=1
#     print(cnt)

# def countRectangles(points):
#     # 使用集合存储唯一的点，去除重复点
#     points_set = set(points)
#     rectangles = set()  # 用来存储已找到的矩形（四个顶点）
#
#     # 枚举所有点对
#     rectangle_count = 0
#     points_list = list(points_set)  # 将去重后的点转化为列表进行枚举
#     n = len(points_list)
#
#     for i in range(n):
#         for j in range(i + 1, n):
#             # 获取点对 (x1, y1) 和 (x2, y2)
#             x1, y1 = points_list[i]
#             x2, y2 = points_list[j]
#
#             # 两点可以构成矩形的对角线，要求x1 != x2 且 y1 != y2
#             if x1 != x2 and y1 != y2:
#                 # 检查另外两个点 (x1, y2) 和 (x2, y1) 是否在集合中
#                 if (x1, y2) in points_set and (x2, y1) in points_set:
#                     # 找到一个矩形，确保矩形四个顶点是唯一的
#                     # 我们将四个点按一定顺序排序，避免矩形重复计数
#                     rectangle = frozenset([(x1, y1), (x2, y2), (x1, y2), (x2, y1)])
#                     if rectangle not in rectangles:
#                         rectangles.add(rectangle)
#                         rectangle_count += 1
#
#     return rectangle_count
#
#
# def main():
#     # 读取测试数据
#     T = int(input())  # 测试数据组数
#     for _ in range(T):
#         n = int(input())  # 每组数据的点的数量
#         points = []
#         for _ in range(n):
#             x, y = map(int, input().split())
#             points.append((x, y))
#
#         # 计算当前测试数据的矩形个数
#         print(countRectangles(points))
#
#
# if __name__ == "__main__":
#     main()

# while True:
#     n=int(input())
#     if n==0:
#         break
#     s=[[' ']*(2**(n+1)) for _ in range(2**n)]
#     a,b=len(s)-1,0
#     def dr(s,n,x,y):
#         if n==1:
#             s[x][y:y+4]="/__\\"
#             s[x-1][y+1:y+3]='/\\'
#             return
#         c,d=x,y+2**n
#         e,f=x-2**(n-1),y+2**(n-1)
#         dr(s,n-1,x,y)
#         dr(s,n-1,c,d)
#         dr(s,n-1,e,f)
#
#     dr(s,n,a,b)
#     for i in s:
#         l=''.join(i)
#         l.rstrip()
#         print(l)
#     print()

####################################################
# from heapq import heappop,heappush
# n=int(input())
# a,b=[],[]
# cnt=0
# for i in range(n):
#     ai,bi=map(int,input().split())
#     heappush(a,(ai,i))
#     heappush(b,(-bi,i))
#     while a[0][0]>b[0][0] and a[0][1]<-b[0][1]:
#         heappop(a)
#         heappop(b)
#     cnt=max(cnt,len(a))
# print(cnt)
# dx,dy=[-1,0,1,-1,0,1,-1,0,1],[-1,-1,-1,0,0,0,1,1,1]
# n,k=map(int,input().split())
# s=[[0]*n for _ in range(n)]
# cnt=0
# def dfs(s,n,k):
#     global cnt
#     if k==0:
#         cnt+=1
#         return
#     for i in range(n):
#         for j in range(n):
#             if not s[i][j]:
#                 for p in range(9):
#                     if 0<=i+dx[p]<n and 0<=j+dy[p]<n:
#                         s[i+dx[p]][j+dy[p]]+=1
#                 dfs(s,n,k-1)
#                 for p in range(9):
#                     if 0 <= i + dx[p] < n and 0 <= j + dy[p] < n:
#                         s[i+dx[p]][j+dy[p]]-=1
# dfs(s,n,k)
# if k==1:
#     print(cnt)
# else:
#     print(cnt//factorial(k))


from collections import deque
from typing import List

an,am=map(int,input().split())
a=[]
for _ in range(an): a.append(list(map(int,input().split())))
bn,bm=map(int,input().split())
b=[]
for _ in range(bn): b.append(list(map(int,input().split())))
cn,cm=map(int,input().split())
c=[]
for _ in range(cn): c.append(list(map(int,input().split())))
if am!=bn or an!=cn or bm!=cm:
    print('Error!')
else:
    for i in range(an):
        for j in range(bm):
            c[i][j]+=sum(a[i][k]*b[k][j] for k in range(am))
    for i in c:
        print(*i)

