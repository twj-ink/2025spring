def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


def union(x, y):
    xp, yp = find(x), find(y)
    if xp == yp:
        return

    if size[xp] > size[yp]:
        parent[yp] = xp
        size[yp] = size[xp]
    elif size[xp] <= size[yp]:
        parent[xp] = yp
        size[xp] = size[yp]


n, m = map(int, input().split())
# parent=[int(i) for i in range(n+1)]
parent = list(range(n + 1))
size = list(map(int, input().split()))
size = [0] + size
for _ in range(m):
    a, b = map(int, input().split())
    union(a, b)

classes = [size[i] for i in range(1, n + 1) if i == parent[i]]
print(len(classes))
classes.sort(reverse=True)
print(*classes)