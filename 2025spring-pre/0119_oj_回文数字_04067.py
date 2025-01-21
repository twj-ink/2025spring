# import queue
# s=input()
# q=queue.Queue(maxsize=)
while True:
    try:
        s=input()
        print(['NO','YES'][s[:len(s)//2][::-1]==s[len(s)//2+1:] if len(s)&1 else s[:len(s)//2][::-1]==s[len(s)//2:]])
    except EOFError:
        break