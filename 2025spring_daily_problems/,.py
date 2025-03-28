def seq(n):
    for i in range(n):
        yield i

n=2
iter_=seq(n)
print(next(iter_))
print(next(iter_))
print(next(iter_))
