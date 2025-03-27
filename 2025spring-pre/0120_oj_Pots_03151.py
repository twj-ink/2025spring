# from collections import deque
# def fill(i,vol,s):
#     vol[i]=s[i]
# def pour(i,j,vol,s):
#     if vol[i]<=s[j]-vol[j]:
#         vol[j]+=vol[i]
#         vol[i]=0
#     else:
#         vol[i]-=s[j]-vol[j]
#         vol[j]=s[j]
# def drop(i,vol,s):
#     vol[i]=0
#
# s=list(map(int,input().split()))
# a=b=0
# def bfs(a,b,s,step,path):
#     q=deque()
#     q.append((a,b,path))
#     inq=set()
#     inq.add((a,b))
#     while q:
#         for _ in range(len(q)):
#             a,b,ppath=q.popleft()
#             if a==s[2] or b==s[2]:
#                 return (step,ppath)
#             for i in range(2):
#                 curr=[a,b]
#                 path=ppath
#                 fill(i,curr,s)
#                 if (curr[0],curr[1]) not in inq:
#                     path+=f'FILL({i+1}) '
#                     inq.add((curr[0],curr[1]))
#                     q.append((curr[0],curr[1],path))
#
#                 curr=[a,b]
#                 path=ppath
#                 pour(i,1-i,curr,s)
#                 if (curr[0],curr[1]) not in inq:
#                     path+=f'POUR({i+1},{2 - i}) '
#                     inq.add((curr[0],curr[1]))
#                     q.append((curr[0],curr[1],path))
#
#                 curr=[a,b]
#                 path=ppath
#                 drop(i,curr,s)
#                 if (curr[0],curr[1]) not in inq:
#                     path+=f'DROP({1+i}) '
#                     inq.add((curr[0],curr[1]))
#                     q.append((curr[0],curr[1],path))
#         step+=1
#     return 'impossible'
#
# res=bfs(0,0,s,0,'')
# if res!='impossible':
#     print(res[0])
#     ans=list(res[1].rstrip().split())
#     for process in ans:
#         print(process)
# else:
#     print(res)
from collections import deque

def fill(i, vol, s):
    vol[i] = s[i]

def pour(i, j, vol, s):
    transfer = min(vol[i], s[j] - vol[j])
    vol[i] -= transfer
    vol[j] += transfer

def drop(i, vol, s):
    vol[i] = 0

def bfs(a, b, s):
    q = deque([(a, b, "")])  # 队列中存储当前状态和路径
    inq = set([(a, b)])  # 记录访问过的状态
    step = 0

    while q:
        for _ in range(len(q)):
            a, b, path = q.popleft()
            if a == s[2] or b == s[2]:  # 如果达到目标水量
                return step, path.strip().split()

            for i in range(2):  # 遍历两个pots和三个方法：FILL, DROP, POUR
                for action in ["fill", "drop", "pour"]:
                    curr = [a, b]
                    new_path = path

                    if action == "fill":
                        fill(i, curr, s)
                        new_action = f"FILL({i + 1})"
                    elif action == "drop":
                        drop(i, curr, s)
                        new_action = f"DROP({i + 1})"
                    elif action == "pour":
                        pour(i, 1 - i, curr, s)
                        new_action = f"POUR({i + 1},{2 - i})"

                    # 检查状态是否已经访问过
                    if tuple(curr) not in inq:
                        inq.add(tuple(curr))
                        q.append((curr[0], curr[1], f"{new_path} {new_action}"))

        step += 1

    return "impossible"

# 输入和处理
s = list(map(int, input().split()))
result = bfs(0, 0, s)

# 输出结果
if result == "impossible":
    print(result)
else:
    steps, operations = result
    print(steps)
    for op in operations:
        print(op)
