import numpy as np

t = np.linspace(-1, 1, 10)
print(t)
y = np.heaviside(t,0.5)

print(y)