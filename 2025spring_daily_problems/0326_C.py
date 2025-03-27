exit = None
def times(n):
    x=0
    while True:
        yield x
        x+=n
n,m = map(int,input().split())
seq = times(n)
if str(type(seq)) == "<class 'generator'>":
    i = 0
    for x in seq:
        print(x)
        i += 1
        if i == m:
            break