def accfunc(f):
    #pylint:skip-file
    funs=[f]
    def inner(ff=None):
        if ff is None:
            def execute(n):
                for fun in funs:
                    n=fun(n)
                return n
            return execute
        funs.append(ff)
        return inner
    return inner

def f1(x):
    return x + 1
def f2(x):
    return x * x
def f3(x):
    return x + x
def f4(x):
    return x*3
def f5(x):
    return x-4

while True:
    try:
        s = input()
        n = int(input())
        s = s.split()
        k = accfunc
        for x in s:
            # print(type(eval(x)))
            k = k(eval(x))
        print(k()(n))
    except EOFError:  #读到 eof产生异常
        break