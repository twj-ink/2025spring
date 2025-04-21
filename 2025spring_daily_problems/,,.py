# n=int(input())
# k=int(input())
# s=list(map(int,input().split()))
# dp=[[-float('inf')]*(k+1) for _ in range(n+1)]
# for i in range(k+1): dp[0][i]=0
# for i in range(1,n+1): dp[i][0]=0; dp[i][1]=s[i-1]
# for i in range(1,n+1):
#     for j in range(1,k+1):
#         for kk in range(1,i):
#             if dp[i-kk][j-1]!=-float('inf'):
#                 dp[i][j] = max(dp[i][j], s[kk-1]+dp[i-kk][j-1])
#
# print(dp[-1][-1])

class TrieNode:
    def __init__(self):
        self.children = dict()
        self.cnt = 1
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        cur = self.root
        for ch in word:
            if ch not in cur.children:
                cur.children[ch] = TrieNode()
            cur = cur.children[ch]
            cur.cnt += 1
        cur.is_end = True

    def get_unique_prefix(self, word):
        cur = self.root
        pre = ''
        for ch in word:
            pre += ch
            cur = cur.children[ch]
            if cur.cnt == 1:
                return pre
        return pre

words=[]
while True:
    try:
        word = input()
        words.append(word)
    except EOFError:
        break

trie = Trie()
for word in words:
    trie.insert(word)

for word in words:
    print(trie.get_unique_prefix(word))