# -*- coding: utf-8 -*-
"""CH04_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tBjazHGO0_KYBM0BRcbdPfBKbGnVKJ5g
"""

import matplotlib.pyplot as plt

x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 100, 200]

plt.plot(x1, y1, color='blue')
plt.plot(x2, y2, color='red')

plt.show()

plt.subplot(2, 1, 2)
plt.plot(x1,y1)
plt.title('data1')

plt.subplot(1, 2, 2)
plt.plot(x2,y2)
plt.title('data2')

fig, axe1 = plt.subplots(nrows = 1, ncols = 2)
axe1[0].plot(x1, y1, color = 'blue')
axe1[1].plot(x1, y1, color = 'red')

fig, axe1 = plt.subplots()
axe2 = axe1.twinx()
axe1.plot(x1, y1, color = 'blue', label = 'data1')
axe2.plot(x2, y2, color = 'red', label = 'data2')

axe1.set_xlabel('x', fontsize = 15)
axe1.set_xlabel('y1', fontsize = 15)
axe1.set_xlabel('y2', fontsize = 15)