class Solution:
    def maxArea(self, s: List[int]) -> int:
        n=len(s)
        i,j=0,n-1
        ans=min(s[i],s[j])*(j-i)
        while i<j:
            if s[i]<s[j]:
                i+=1
                ans=max(ans,min(s[i],s[j])*(j-i))
            else:
                j-=1
                ans=max(ans,min(s[i],s[j])*(j-i))
        return ans

#cpp
# #include <algorithm>
# class Solution {
# public:
#     int maxArea(vector<int>& height) {
#         int n=height.size();
#         int i=0;
#         int j=n-1;
#         int ans=std::min(height[j],height[i])*(j-i);
#         while (i<j)
#         {
#             if (height[i]<height[j]){
#                 i++;
#             } else {
#                 j--;
#             }
#             ans=std::max(ans,std::min(height[i],height[j])*(j-i));
#         }
#         return ans;
#     }
# };