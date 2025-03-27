def mergeSort(s):
    n=len(s)
    if n<=1:
        return s
    mid=n//2
    left=mergeSort(s[:mid])
    right=mergeSort(s[mid:])
    return merge(left,right)

def merge(l,r):
    global res
    ans=[]
    i,j=0,0
    while i<len(l) and j<len(r):
        if l[i]<r[j]:
            ans.append(l[i])
            i+=1
        else:
            ans.append(r[j])
            res+=(len(l)-i)
            j+=1
    ans.extend(l[i:])
    ans.extend(r[j:])
    return ans

while True:
    n=int(input())
    if n==0:
        break
    s=[]
    for _ in range(n):
        s.append(int(input()))
    res=0
    mergeSort(s)
    print(res)
