from math import gcd
class Fraction:
    def __init__(self,top,bot):
        d=gcd(top,bot)
        self.top=top//d
        self.bottom=bot//d

    def __add__(self,b):
        new_top=self.top*b.bottom+self.bottom*b.top
        new_bottom=self.bottom*b.bottom
        return Fraction(new_top,new_bottom)

    def __str__(self):
        return str(self.top)+'/'+str(self.bottom)

    # def show(self):
    #     print(str(self.top)+'/'+str(self.bottom))

t1,b1,t2,b2=map(int,input().split())
a,b=Fraction(t1,b1),Fraction(t2,b2)
print(a+b)
# ans=a+b
# ans.show()