import matplotlib.pyplot as plt
x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 4, 7]

# 그래프에 두개 표현하기
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

# 범례 위치 / 크기 설정
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