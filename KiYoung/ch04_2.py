# -*- coding: utf-8 -*-
"""CH04_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1p-ClIKA-aw_hT7DGYHu_YY4juk66urF2
"""

import matplotlib.pyplot as plt
x1 = [1, 3, 4]
y1 = [1, 2, 3]
x2 = [1, 5, 7]
y2 = [1, 7, 8]

# plt.plot(x1, y1, color='blue')
# plt.plot(x2, y2, color='red')

# plt.plot(x1, y1, 'b', x2, y2, 'r')

# 범례추가
# plt.plot(x1, y1, color='blue', label='data1')
# plt.plot(x2, y2, color='red', label='data2')
# plt.legend()

# plt.plot(x1, y1, color='blue')
# plt.plot(x2, y2, color='red')
# plt.legend(['data1', 'data2'])

# plt.plot(x1, y1, color='blue')
# plt.plot(x2, y2, color='red')
# plt.legend(['data1', 'data2'], loc='upper right')

# plt.plot(x1, y1, color='blue')
# plt.plot(x2, y2, color='red')
# plt.legend(['data1', 'data2'], fontsize=20)

x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 100, 200]

plt.plot(x1, y1, color='blue')
plt.plot(x2, y2, color='red')

plt.show()