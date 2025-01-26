class MinStack:

    def __init__(self):
        self.stack=[]
        self.b=[]

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.b:
            self.b.append(val)
        else:
            self.b.append(min(self.b[-1],val))

    def pop(self) -> None:
        self.stack.pop()
        self.b.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.b[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()