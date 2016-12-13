# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def step(x):
  return np.array(x > 0, dtype = np.int)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = step(x)
y2 = sigmoid(x)

plt.plot(x, y1, label = 'step')
plt.plot(x, y2, linestyle = '--', label = 'sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.title('step & sigmoid')
plt.ylim(-0.1, 1.1)
plt.show()
