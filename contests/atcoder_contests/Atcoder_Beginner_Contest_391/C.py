# WA & TLE
# from collections import defaultdict,deque
#
# guiji=defaultdict(deque)
# cnt=defaultdict(int)
#
# n,q=map(int,input().split())
# for _ in range(q):
#     s=input()
#     if s!='2':
#         _,a,b=map(int,s.split())
#         guiji[a].append(b)
#         cnt[a]=max(cnt[a]-1,-1)
#         cnt[b]+=1
#     else:
#         num=0
#         for k,v in guiji.items():
#             while len(v)>1:
#                 cut=v.popleft()
#                 cnt[cut]=max(cnt[cut]-1,-1)
#         for v in cnt.values():
#             num+=1 if v>=1 else 0
#         print(num)