def mysum(x):
    def inner(y=None):
        if y is not None:
            return mysum(x+y)
        return x
    return inner

k = mysum(1)(2)(3)(4)
k2 = mysum(10)(20)

print(k())
print(k2())

def exec(g,i) : #call g for i times
    if i == 1 :
        g()
    else:
        g()
        exec(g,i-1)
while True:
    try:
        s = input()
        s = s.split()
        k = mysum
        for x in s:
            k = k(int(x))
        exec(k,int(s[0]))
        print(k())
    except:  #读到 eof产生异常
        break