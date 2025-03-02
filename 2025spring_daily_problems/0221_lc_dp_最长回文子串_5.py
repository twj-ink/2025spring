class Solution:
    def longestPalindrome(self, s: str) -> str:
        # dp[i][j] 表示从i到j是回文的
        # dp[i][j]=dp[i+1][j-1] and (s[i]==s[j])
        n = len(s)
        maxl = 1;
        begin = 0
        dp = [[False] * n for _ in range(n)]

        # 单个字符为true
        for i in range(n):
            dp[i][i] = True

        # 枚举长度
        for L in range(2, n + 1):
            for i in range(n):
                j = L + i - 1
                if j >= n:
                    break

                if s[i] != s[j]:
                    dp[i][j] = False
                elif s[i] == s[j] and j - i < 3:
                    dp[i][j] = True
                elif s[i] == s[j] and j - i >= 3:
                    dp[i][j] = dp[i + 1][j - 1]

                if dp[i][j] and L > maxl:
                    maxl = L
                    begin = i
        return s[begin:begin + maxl]

    #     class Solution:
    # def expand(self,s,l,r):
    #     while l>=0 and r<len(s) and s[l]==s[r]:
    #         l-=1
    #         r+=1
    #     return l+1,r-1

    # def longestPalindrome(self, s: str) -> str:
    #     start,end=0,0
    #     for i in range(len(s)):
    #         l1,r1=self.expand(s,i,i)
    #         l2,r2=self.expand(s,i,i+1)
    #         if r1-l1>end-start:
    #             start,end=l1,r1
    #         if r2-l2>end-start:
    #             start,end=l2,r2
    #     return s[start:end+1]
    #     # n=len(s)
    #     # if n==1:
    #     #     return ''.join(s)
    #     # if n==2:
    #     #     return (''.join(s) if s[0]==s[1] else s[0])

    #     # d1={};d2={0:0}
    #     # for i in range(1,n-1):
    #     #     half=1
    #     #     while i-half>=0 and i+half<=n-1 and s[i-half:i+half+1][::-1]==s[i-half:i+half+1]:
    #     #         half+=1
    #     #     d1[i]=2*(half-1)+1

    #     # for i in range(n-1):
    #     #     if s[i]==s[i+1]:
    #     #         half=1
    #     #         while i-half>=0 and i+1+half<=n-1 and s[i-half:i+1+half+1][::-1]==s[i-half:i+1+half+1]:
    #     #             half+=1
    #     #         d2[i]=2*(half-1)+2

    #     # index1=max(d1,key=d1.get)
    #     # length1=d1[index1]
    #     # index2=max(d2,key=d2.get)
    #     # length2=d2[index2]
    #     # if length1>length2:
    #     #     half=(length1-1)//2
    #     #     return ''.join(s[index1-half:index1+half+1])
    #     # half=(length2-2)//2
    #     # return ''.join(s[index2-half:index2+1+half+1])
