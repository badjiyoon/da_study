# CH04_03. subplot / subplots 를 이용한 여러 개의 차트 그리기

import matplotlib.pyplot as plt

#문제 : data1(x : 1, 2, 3; y : 1, 2, 3) 과 data2(x : 1, 2, 3; y : 1, 100, 200) 을 그래프로 출력하시오.
x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 100, 200]

#subplot을 이용한 해결
#subplot: 행의 수, 열의 수, 해당 그래프가 그려질 위치

plt.subplot(1, 2, 1) 
plt.plot(x1, y1)
plt.title('data1')

plt.subplot(1, 2, 2)
plt.plot(x2, y2)
plt.title('data2')
plt.show() #표 그리기랑 비슷, 1행 2열

plt.subplot(2, 1, 1)
plt.plot(x1, y1)
plt.title('data1')

plt.subplot(2, 1, 2)
plt.plot(x2, y2)
plt.title('data2')
plt.show() #표 그리기랑 비슷, 2행 1열

#타이틀과 인덱스가 겹칠 경우
#tight_layout 을 이용하면 레이아웃이 자동으로 설정됨

plt.subplot(2, 1, 1)
plt.plot(x1, y1)
plt.title('data1')

plt.subplot(2, 1, 2)
plt.plot(x2, y2)
plt.title('data2')
plt.tight_layout()
plt.show() 

#subplots 를 이용한 해결
fig1, axe1 = plt.subplots(nrows = 1, ncols = 2) #도화지를 준비하는 개념
fig1.show()
axe1

'''
fig, axe1 모두 내가 정하는 이름, 변수
첫 번째 위치(fig)는 도화지, 두 번째 위치(axe1)는 그 안에 들어가는 데이터
array([<AxesSubplot: >, <AxesSubplot: >], dtype=object) 무슨 뜻일까?
'''

fig, axe1 = plt.subplots(nrows = 1, ncols = 2)
axe1[0].plot(x1, y1, color = 'blue') #[0] = 첫 번째에 plot 그려주기
axe1[1].plot(x2, y2, color = 'red') #[1] = 두 번째에 plot 그려주기
plt.show()

#문제 : data1(x : 1, 2, 3; y : 1, 2, 3) 과 data2(x : 1, 2, 3; y : 1, 50, 200) 을 그래프로 출력하시오.
x1 = [1, 2, 3]
y1 = [1, 2, 3]
x2 = [1, 2, 3]
y2 = [1, 50, 200]

fig, axe1 = plt.subplots() 

axe2 = axe1.twinx() #axe2는 axe1을 복사하여 그래프 2개 그려짐, fig는 1개인데 axe가 2개
axe1.plot(x1, y1, color = 'red', label = 'data1') 
axe2.plot(x2, y2, color = 'blue', label = 'data2')

axe1.set_xlabel('x', fontsize = 15)
axe1.set_ylabel('y1', fontsize = 15)
axe2.set_ylabel('y2', fontsize = 15) 

'''
subplots = subplot들이 하나의 좌표평면에 표시
label 이름을 붙일 때 set_xlabel, set_ylabel과 같이 set을 표시
'''

plt.legend(['data1', 'data2']) #레전드 넣어도 범례에 data1만 표시됨
plt.show()

#y 축을 두 개 가진 차트에서 범례 표시하기

fig, axe1 = plt.subplots()

axe2 = axe1.twinx()
chart1 = axe1.plot(x1, y1, color = 'red') #chart1이라는 이름으로 저장
chart2 = axe2.plot(x2, y2, color = 'blue') #chart2라는 이름으로 저장

axe1.set_xlabel('x', fontsize = 15)
axe1.set_ylabel('y1', fontsize = 15)
axe2.set_ylabel('y2', fontsize = 15)

chart = chart1 + chart2 #chart1과 chart2 둘을 더하여 하나의 chart로
axe1.legend(chart, ['data1', 'data2']) #legend 옵션인듯. (범례를 넣을 그래프 이름, 범례명)
plt.show()