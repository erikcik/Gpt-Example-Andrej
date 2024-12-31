import numpy as np
import matplotlib.pyplot as plt

A = 10
test = lambda x: A + (x**2 - A * np.cos(2 * np.pi * x))
# Derivative of the Rastrigin function
testDer = lambda x: 2*x + 2*A*np.pi*np.sin(2*np.pi*x)



x = np.linspace(-10, 10, 101) # creates evenly spaced values for the given inputs
# print(x)
y = test(x) # you can basically put list inside of the function just like this even thoug you didnt declare any type whatsoever
# print(y)

steps = []
startX = 4
for i in range(50):
    steps.append(startX)
    grad = testDer(startX)
    print(f"Iteration {i+1}:")
    print(f"  Start X: {startX:.4f}")
    print(f"  Gradient: {grad:.4f}")
    startX = startX - 0.05 * grad
    print(f"  Updated Start X: {startX:.4f}")
    print(f"  Step {i+1}: {int(steps[-1])}\n")


plt.plot(x, y, label='f(x) = x^2', color='blue')
plt.scatter(steps, test(np.array(steps)), color='red')
plt.scatter(steps[49], test(np.array(steps[49])), color='green')
plt.savefig("gradient_descent_plot.png")
print("Plot saved as 'gradient_descent_plot.png'")


