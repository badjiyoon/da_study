import matplotlib.pyplot as plt

x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 100, 200]
"""
# 행의 수 , 열의 수, 해당 그래프가 그려질 위치
plt.subplot(1, 2, 1)
plt.plot(x1, y1)
plt.title('data1')

plt.subplot(1, 2, 2)
plt.plot(x2, y2)
plt.title('data2')

# 상 하 구조
plt.subplot(2, 1, 1)
plt.plot(x1, y1)
plt.title('data1')

plt.subplot(2, 1, 2)
plt.plot(x2, y2)
plt.title('ata2')
# tight_layout 이용하여 자동설정
plt.subplot(2, 1, 1)
plt.plot(x1, y1)
plt.title('data1')

plt.subplot(2, 1, 2)
plt.plot(x2, y2)
plt.title('ata2')
plt.tight_layout()
# subplots를 이용한 처리
fig, axe1 = plt.subplots(nrows=1, ncols=2)
axe1[0].plot(x1, y1, color='blue')
axe1[1].plot(x2, y2, color='red')
"""
x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 50, 200]

fig, axe1 = plt.subplots()
axe2 = axe1.twinx()
chart1 = axe1.plot(x1, y1, color='red', label='data1')
chart2 = axe2.plot(x2, y2, color='blue', label='data2')
axe1.set_xlabel('x', fontsize=15)
axe1.set_ylabel('y1', fontsize=15)
axe2.set_ylabel('y2', fontsize=15)

# 차트를 더해준다.
chart = chart1 + chart2
# 범례처리
axe1.legend(chart, ['data1', 'data2'])

plt.show()