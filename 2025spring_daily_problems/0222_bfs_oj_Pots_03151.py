from collections import deque

def bfs(s,path,step):
    def fill(i,curr,s):
        curr[i-1]=s[i-1]
    def drop(i,curr,s):
        curr[i-1]=0
    def pour(i,j,curr,s):
        rest=s[2-i]-curr[2-i]
        if rest>=curr[i-1]:
            curr[2-i]+=curr[i-1]
            curr[i-1]=0
        else:
            curr[i-1]-=rest
            curr[2-i]=s[2-i]
    q=deque()
    q.append((0,0,path,step))
    inq=set()
    inq.add((0,0))
    while q:
        for _ in range(len(q)):
            a,b,path,step=q.popleft()
            # print(a,b,path)
            if a==s[2] or b==s[2]:
                return step,path[:]
            for i in range(1,3):
                for action in ['fill','drop','pour']:
                    curr=[a,b]
                    new=path[:]
                    if action=='fill':
                        fill(i,curr,s)
                        new.append(f'FILL({i})')
                        # q.append((curr[0],curr[1],path,step+1))
                    elif action=='drop':
                        drop(i,curr,s)
                        new.append(f'DROP({i})')
                        # q.append((curr[0],curr[1],path,step+1))
                    elif action=='pour':
                        # if curr==[0,2]:
                        #     print(111,i,3-i)
                        pour(i,3-i,curr,s)
                        new.append(f'POUR({i},{3-i})')

                    if tuple(curr) not in inq:
                        inq.add(tuple(curr))
                        q.append((curr[0],curr[1],new,step+1))
    return 'impossible'


A,B,C=map(int,input().split())
s=[A,B,C]
path,step=[],0
ans=bfs(s,path,step)
if ans=='impossible':
    print(ans)
else:
    step,path=ans
    print(step)
    for i in path:
        print(i)