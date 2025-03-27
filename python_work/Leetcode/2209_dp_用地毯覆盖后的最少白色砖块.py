class Solution:
    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        floor=list(map(int,floor))
        @cache
        def dfs(i,j): # 到第j个砖块用i个地毯覆盖的最少数目
            if j<i*carpetLen:
                return 0
            if i==0:
                return dfs(i,j-1)+floor[j] #最后一个砖块不覆盖
            return min(dfs(i,j-1)+floor[j],dfs(i-1,j-carpetLen))
        return dfs(numCarpets,len(floor)-1)