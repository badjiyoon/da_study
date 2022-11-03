import matplotlib.pyplot as plt

# 단일 차트 그리기
x = [1, 2, 3]
y = [4, 5, 6]

plt.figure()
plt.plot(x, y)
# plt.show()

# lineplot은 y값만 있으면 그림을 그릴 수 있음
# 이 때 x의 값은 0, 1, 2,... 순서로 자동 지정됨

plt.plot(y)
# plt.show()

plt.plot(x, y, linewidth=10)
plt.plot(x, y, color='red')
plt.plot(x, y, marker='o')
plt.plot(x, y, marker='o', markersize=10)
plt.plot(x, y, linestyle=':')
plt.plot(x, y, ':')
plt.plot(x, y, 'ro')

# 그래프 명
plt.plot(x, y)
plt.title('title')
plt.title('title', fontsize=20)

# X 축명, y 축명
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20)

# 그리드 설정
plt.plot(x, y)
plt.grid(True)
# 그리드 설정(x 축만)
plt.plot(x, y)
plt.grid(True, axis='x')
# 그리드 설정(y 축만)
plt.plot(x, y)
plt.grid(True, axis='y')

# x범위, y범위 설정
plt.plot(x, y)
plt.xlim([1, 2])
plt.ylim([4, 5])

# axis를 사용한 범위 설정
plt.plot(x, y)
plt.axix([1, 2, 4, 5])
# 눈금 글꼴 크기 변경
plt.plot(x, y)
plt.xticks(fontsize=15)
plt.yticks(fontsize=20)
# 그래프에 텍스트 삽입
plt.plot(x, y)
plt.text(2, 5, 'text', fontsize='20')
# 한글 폰트 사용하기
plt.plot(x)
plt.title('차트 명')
plt.show()
