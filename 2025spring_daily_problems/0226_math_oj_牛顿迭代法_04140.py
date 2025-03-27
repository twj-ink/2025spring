def f(x):
    return x ** 3 - 5 * x ** 2 + 10 * x - 80


def df(x):
    return 3 * x ** 2 - 10 * x + 10


def newton_method(f, df, x0, tol=1e-9, max_iter=1000):
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        # 如果导数为零，停止迭代
        if dfx == 0:
            return None

        # 计算下一次迭代
        x_new = x - fx / dfx

        # 如果误差小于容忍度，则返回结果
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    return None


# 初始猜测值为 5
root = newton_method(f, df, 5.0)
if root is not None:
    print(f"{root:.9f}")
else:
    print("Error!")
