class Solution:
    def findValidPair(self, s: str) -> str:
        a = []
        for i in s:
            a.append(int(i))
        c = Counter(a)
        for i in range(len(s) - 1):
            if s[i] != s[i + 1] and c[int(s[i])] == int(s[i]) and c[int(s[i + 1])] == int(s[i + 1]):
                return s[i] + s[i + 1]

        return ''