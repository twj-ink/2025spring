from collections import deque,defaultdict

def build(words):
    buckets = defaultdict(list)
    for word in words:
        for i in range(4):
            curr = word[:i] + '*' + word[i + 1:]
            buckets[curr].append(word)

    g = defaultdict(list)
    for bucket in buckets.values():
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a,b = bucket[i], bucket[j]
                g[a].append(b)
                g[b].append(a)

    return g

def bfs(g, start, end):
    q = deque([start])
    prev = {start:None}
    found = False

    while q:
        if found:
            break
        for _ in range(len(q)):
            curr = q.popleft()
            if curr == end:
                found = True
                break
            for next in g[curr]:
                if next not in prev:
                    q.append(next)
                    prev[next] = curr

    if not found:
        return 'NO'
    else:
        path=[]
        cur=end
        while cur:
            path.append(cur)
            cur = prev[cur]
        return ' '.join(path[::-1])
n=int(input())
words=[input() for _ in range(n)]
start, end = input().split()
g = build(words)
print(bfs(g, start, end))