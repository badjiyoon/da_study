#CH05_06. numpy를 이용한 행렬 연산

import numpy as np

#행렬 생성
a = np.array([[1, 2], [3, 4]])
'''
리스트: 
[[1, 2], [3, 4]]
행렬: 
[1, 2]
[3, 4]
'''

#전치 행렬: 행과 열을 바꾼 행렬
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.T) #행렬 이름.T : 전치 행렬

#행렬 차원 확인
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape) #(2, 3) -> 2행 3열

#행렬 형태 변경
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.reshape(a, (3, 2)) #a라는 행렬을 3행 2열로 reshape

print(b.shape)
print(b)

#배열간 사칙연산
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[2, 3, 4], [5, 6, 7]])

print(a + b)
print(a + 3)

print(a - b)
print(a - 2)

print(a * b) #곱셈은 내적이 아니고 각 데이터들을 곱해줌
print(a * 2)

print(a / b)
print(a / 2)

#행렬의 형태가 다른 경우에는 불가
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[2, 3], [5, 6]])

print(a + b) #행렬의 크기가 달라 연산 불가

#행렬의 곱셈
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.dot(a, b)) #dot연산: 내적

a = np.array([[1, 2, 3], 
              [4, 5, 6]])
b = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print(np.dot(a, b))