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
##### Codeforces Round 997 #####
#### A ####

#### B ####

#### C ####
