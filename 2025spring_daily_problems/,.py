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

print(9**(1/3))