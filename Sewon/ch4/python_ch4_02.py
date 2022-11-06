# CH04_02. 여러 데이터를 한 차트에 그리기

import matplotlib.pyplot as plt

#문제 : data1(x : 1, 2, 3; y : 1, 2, 3) 과 data2(x : 1, 2, 3; y : 1, 4, 7) 을 그래프로 출력하시오.

x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 4, 7]

plt.plot(x1, y1, color = 'blue')
plt.plot(x2, y2, color = 'red')
plt.show()

plt.plot(x1, y1, 'b', x2, y2, 'r')
plt.show()

#범례 추가
#문제 : data1(x : 1, 2, 3; y : 1, 2, 3) 과 data2(x : 1, 2, 3; y : 1, 4, 7) 을 그래프로 출력하시오.
x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 4, 7]

plt.plot(x1, y1, color = 'blue', label = 'data1') #color, label 옵션, label은 그래프에 이름을 붙여준 것
plt.plot(x2, y2, color = 'red', label = 'data2')
plt.legend() #legend가 있어야 범례가 표시됨
plt.show()

plt.plot(x1, y1, color = 'blue')
plt.plot(x2, y2, color = 'red')
plt.legend(['data1', 'data2']) #코드로 나타낸 줄 순서대로 data1, data2 라벨링됨
plt.show()

#범례 위치 설정
plt.plot(x1, y1, color = 'blue')
plt.plot(x2, y2, color = 'red')
plt.legend(['data1', 'data2'], loc = 'upper right') #locatation = loc, 범례의 위치
plt.show()

#범례 폰트 사이즈 변경
plt.plot(x1, y1, color = 'blue')
plt.plot(x2, y2, color = 'red')
plt.legend(['data1', 'data2'], fontsize = 20)
plt.show()

#문제 : data1(x : 1, 2, 3; y : 1, 2, 3) 과 data2(x : 1, 2, 3; y : 1, 100, 200) 을 그래프로 출력하시오.
x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 100, 200]

plt.plot(x1, y1, color = 'blue')
plt.plot(x2, y2, color = 'red')
plt.show()