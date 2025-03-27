def f(x):
    return x**3-5*x**2+10*x-80
def df(x):
    return 3*x**2-10*x+10

def newton_method(f,df,x0,tol=1e-9,max_iter=10000):
    x=x0
    for i in range(max_iter):
        fx=f(x)
        dfx=df(x)
        if dfx==0:
            return None
        x_new=x-fx/dfx
        if abs(x_new-x)<tol:
            return x_new

        x=x_new

    return None

root=newton_method(f,df,1.0)
if root is not None:
    print(f'{root:.9f}')
else:
    print('Error!')
