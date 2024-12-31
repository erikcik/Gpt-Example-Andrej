h = 0.0001

a = 2.0
b = -3.0
c = 10.0

d1 = (a * b) + c
a += h
d2 = (a * b) + c

print((d2 - d1) / h)