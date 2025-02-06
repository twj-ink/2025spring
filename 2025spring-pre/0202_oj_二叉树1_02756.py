s=list(map(int,input().split()))
ans=[[s[0]],[s[1]]]
for i in range(2):
    while s[i]>1:
        ans[i].append(s[i]//2)
        s[i]//=2
ans[0].sort()
ans[1].sort()
for i in range(min(len(ans[0]),len(ans[1]))):
    if ans[0][i]!=ans[1][i]:
        print(ans[0][i-1])
        break
else:
    print(ans[0][-1])