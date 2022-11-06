# CH04_01. matplotlib을 이용한 단일 차트 그리기

#library 호출
import matplotlib.pyplot as plt
import matplotlib as mpl

#문제: x가 1, 2, 3이고 y가 4, 5, 6인 그래프를 그리시오.
x=[1, 2, 3]
y=[4, 5, 6]

plt.figure() #도화지를 만드는 것
plt.plot(x, y) #정의역, 공역을 정하는 것
plt.show() #다른 창 띄우기

plt.plot(x, y)
plt.show()

#lineplot은 y값만 있으면 그림을 그릴 수 있음
#이 때 x의 값은 0, 1, 2, ...의 순서로 자동 지정
plt.plot(y)
plt.show()

#linewidth를 이용한 선 굵기 조정
plt.plot(x, y, linewidth=10)
plt.show()

#color를 통한 선 색상 조정
plt.plot(x, y, color='red') #r만 쳐도 빨간색 나옴
plt.show()

#marker를 이용한 데이터 위치 표시
plt.plot(x, y, marker='o') #동그라미 표시를 하라는 뜻
plt.show()
#matplotlib.org/stable/api/markers_api.html

#선 형태 변경
plt.plot(x, y, linestyle=':') #점선으로 그리기
plt.show()

plt.plot(x, y, 'ro') #빨간색 동그라미로 marker 하겠다는 약어
plt.show()

#그래프 명 title
plt.plot(x, y)
plt.title('title') 
plt.show()

plt.plot(x, y)
plt.title('title', fontsize=20) #폰트 사이즈 커짐
plt.show()

#x축 명, y축 명
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#x축 명, y축 명 폰트 사이즈 변경 label
plt.plot(x, y)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.show()

#그리드 설정
plt.plot(x, y)
plt.grid(True) #좌표평면에 격자무늬
plt.show()

#x축 그리드 -세로줄 grid x
plt.plot(x, y)
plt.grid(True, axis='x') 
plt.show()

#y축 그리드 -가로줄 grid y
plt.plot(x, y)
plt.grid(True, axis='y') 
plt.show()

#x, y 범위 설정 lim
plt.plot(x, y)
plt.xlim([1, 2])
plt.ylim([4, 5])
plt.show()

#axis를 사용한 범위 설정 axis
plt.plot(x, y)
plt.axis([1, 2, 4, 5]) #x 최솟값, x 최댓값, y 최솟값, y 최댓값
plt.show()

#눈금 글꼴 크기 변경 ticks
plt.plot(x, y)
plt.xticks(fontsize=15)
plt.yticks(fontsize=17)
plt.show()

#그래프에 텍스트 삽입
plt.plot(x, y)
plt.text(2, 5, 'text', fontsize='20') #(2, 5)에 text 출력
plt.show()

#한글 폰트 사용하기
plt.plot(x, y)
plt.title('차트명') #한글 안나옴
plt.show()

'''
한글 깔아주기 (아마도 universal?)
아래 코드 실행하고, 런타임 -> 런타임 다시 시작 실행

pip install matplotlib -u
sudo apt-get install -y fonts-nanum
sudo fc-cache -fv
rm ~\.cache/matplotlib -rf

코랩에선 되겠지만 여기선 안됨
'''

sorted([f.name for f in mpl.font_manager.fontManager.ttflist]) #내 컴에 설치된 폰트 보기
plt.rcParams['font.family'] = 'Malgun Gothic' #폰트 설정하기
plt.rcParams['font.family'] = 'NanumGothic'

plt.plot(x, y)
plt.title('차트명') 
plt.show()

plt.rcParams['font.family']=['NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False #뭔지 모르겠음
plt.plot(x, y)
plt.show()

x = [1, 2, 3]
plt.plot(x)
plt.title('차트 명')
plt.show()