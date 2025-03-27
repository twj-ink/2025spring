s=input()
d={'N':'S','E':'W','NE':'SW','NW':'SE'}
dd={v:k for k,v in d.items()}
if s in d:
    print(d[s])
else:
    print(dd[s])