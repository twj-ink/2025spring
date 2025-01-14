s=[]
for _ in range(int(input())):
    line=input()
    if len(line)==3:
        print(s.pop()) if s else print(-1)
    else:
        s.append(int(line.split()[1]))