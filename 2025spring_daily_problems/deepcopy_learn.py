# def deepcopy(a):
#
#     memo = {}  # 用于存储已拷贝的对象，防止循环引用
#
#     obj_id = id(a)  # 获取对象的唯一标识
#     if obj_id in memo:  # 如果已经拷贝过，直接返回
#         return memo[obj_id]
#
#     # 基本数据类型（不可变类型）直接返回
#     if isinstance(a, (int, float, str, bool, type(None))):
#         return a
#
#     # 处理列表
#     if isinstance(a, list):
#         copy_obj = []
#         memo[obj_id] = copy_obj  # 先存入 memo，防止递归调用时无限循环
#         copy_obj.extend(deepcopy(item) for item in a)
#         return copy_obj
#
#     # 处理元组（tuple 需要转换为 tuple）
#     if isinstance(a, tuple):
#         copy_obj = tuple(deepcopy(item) for item in a)
#         memo[obj_id] = copy_obj
#         return copy_obj
#
#     # 处理集合（set）
#     if isinstance(a, set):
#         copy_obj = set(deepcopy(item) for item in a)
#         memo[obj_id] = copy_obj
#         return copy_obj
#
#     # 处理字典
#     if isinstance(a, dict):
#         copy_obj = {}
#         memo[obj_id] = copy_obj
#         for key, value in a.items():
#             copy_obj[deepcopy(key)] = deepcopy(value)
#         return copy_obj
#
#     # 处理自定义类对象（需要用 __dict__ 进行拷贝）
#     if hasattr(a, "__dict__"):
#         copy_obj = a.__class__.__new__(a.__class__)  # 创建一个新的实例
#         memo[obj_id] = copy_obj  # 先存入 memo，防止递归调用
#         copy_obj.__dict__ = deepcopy(a.__dict__)  # 递归拷贝对象属性
#         return copy_obj
#
#     # 其他类型（无法处理）
#     raise TypeError(f"Unsupported type: {type(a)}")
# a = [1,2,[3,[4],5],(6,[7,[8],9])]
# b = deepcopy(a)
# print(b)
# a[2][1].append(400)
# a[3][1][1].append(800)
# print(a)
# print(b)
#
#
from collections import deque
s=[[0]*5 for _ in range(5)]
cnt=0
paths=[]
dx,dy=[0,-1,1,0],[-1,0,0,1]

def canpalce(x,y,s,t):
    if s[x][y]==0:
        s[x][y]=t
        if x==y or x+y==2:
            if all(s[i][i]==t for i in range(3)) or all(s[i][4-i]==t for i in range(3)) or \
                all(s[x][i]==t for i in range(3)) or all(s[i][y]==t for i in range(3)):
                s[x][y]=0
                return False
            s[x][y]=0
            return True
        else:
            if all(s[x][i]==t for i in range(3)) or all(s[i][y]==t for i in range(3)):
                s[x][y]=0
                return False
            s[x][y]=0
            return True
    return False

def dfs(s,x,y,curr,paths,t):
    global cnt

    # print((x,y,t))
    # print(curr)
    if canpalce(x,y,s,t):
        s[x][y]=t
        curr.append((x,y,t))
        if len(curr)==9:
            curr.sort()
            if curr not in paths:
                cnt+=1
                paths.append(curr[:])
                return
        for i in range(4):
            nx,ny=x+dx[i],y+dy[i]
            if 0<=nx<3 and 0<=ny<3 and (nx,ny,3-t) not in curr and canpalce(nx,ny,s,3-t):
                dfs(s,nx,ny,curr,paths,3-t)
            if 0 <= nx < 3 and 0 <= ny < 3 and (nx, ny, t) not in curr and canpalce(nx, ny, s, t):
                dfs(s,nx,ny,curr,paths,t)
        curr.pop()
        s[x][y]=0

for i in range(3):
    for j in range(3):
        dfs(s,i,j,[],paths,1)
print(cnt)
