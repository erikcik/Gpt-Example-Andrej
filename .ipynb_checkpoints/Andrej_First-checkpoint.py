class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self.op = _op
    
    def __repr__(self): # when you try to print the object this function gets called
        return f"The value is {self.data}"

    '''
    gets executed when you call value objects like a + b where a and b are both Value object
    what python does under the hood is a.__add__(b) and a data accessible by self parameter and b accessible by other parameter
    '''
    def __add__(self, other): 
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __mul__(self, other): #basically same with add
        out = Value(self.data * other.data, (self, other), '*')
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a + b
print(d)
e = a * b
print(e)
f = (a * b) + c
#basically (a.__mul__(b)).__add__(c)
print(f)
print(f._prev)
print(e._prev)

