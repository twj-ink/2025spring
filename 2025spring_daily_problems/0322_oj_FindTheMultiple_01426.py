from collections import deque
while True:
    n=int(input())
    if n==0:
        break
    ans='1'
    q=deque()
    q.append(ans)
    inq=set()

    while q:
        for _ in range(len(q)):
            curr = q.popleft()
            num=int(curr)
            if num%n==0:
                print(num)
                break

            for d in ['1','0']:
                new_num = curr + d
                remainder=int(new_num)%n

                if remainder not in inq:
                    inq.add(remainder)
                    q.append(new_num)
