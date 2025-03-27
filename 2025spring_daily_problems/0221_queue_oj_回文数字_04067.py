while True:
    try:
        s=input()
        print(['NO','YES'][s==s[::-1]])
    except EOFError:
        break