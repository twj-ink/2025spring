row,col,dia,xdia=[0]*5,[0]*5,[0]*10,[0]*9
s=[[0]*5 for _ in range(5)]
cnt=0
cnt1,cnt2=13,12
def dfs(i,j,cnt1,cnt2):
    global cnt
    if j==5:
        i+=1
        j=0
    if i==5:
        cnt+=1
        return

    if cnt1:
        row[i]+=1;col[j]+=1;dia[i+j]+=1;xdia[i-j+4]+=1
        if (row[i]==5 or col[j]==5 or dia[i+j]==5 or xdia[i-j+4]==5):
            row[i]-=1;col[j]-=1;dia[i+j]-=1;xdia[i-j+4]-=1
        else:
            dfs(i,j+1,cnt1-1,cnt2)
            row[i] -= 1;
            col[j] -= 1;
            dia[i + j] -= 1;
            xdia[i - j + 4] -= 1
    if cnt2:
        row[i] -= 1;
        col[j] -= 1;
        dia[i + j] -= 1;
        xdia[i - j + 4] -= 1
        if (row[i] == -5 or col[j] == -5 or dia[i + j] == -5 or xdia[i - j + 4] == -5):
            row[i] += 1;
            col[j] += 1;
            dia[i + j] += 1;
            xdia[i - j + 4] += 1
        else:
            dfs(i, j + 1,cnt1,cnt2-1)
            row[i] += 1;
            col[j] += 1;
            dia[i + j] += 1;
            xdia[i - j + 4] += 1



dfs(0,0,13,12)
print(cnt)
