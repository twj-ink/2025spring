import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
s = [int(i) for i in data[index:index+n]]
# n,m=map(int,input().split())
# s=list(map(int,input().split()))
hash_table = [-1] * m
idx = []
for i in s:
    base = 0
    while True:
        pos = (i % m + base ** 2) % m
        if hash_table[pos] in (-1, i):
            hash_table[pos] = i
            idx.append(pos)
            break
        pos = (i % m - base ** 2) % m
        if hash_table[pos] in (-1, i):
            hash_table[pos] = i
            idx.append(pos)
            break
        base += 1
print(*idx)

