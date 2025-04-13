'''
        1
9                1
           8          1
                  10      1
                       6


'''
if all((k//2)+1 == (k+1)//2 for k in range(1000)):
    print(1)