import numpy as np

a = np.array([[1, 2], [3, 4]])
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.T)
# 행렬 자원 확인
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)
b = np.reshape(a, (3, 2))
print(b.shape)
print(b)
# 배열간 사칙연산=
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[2, 3, 4], [5, 6, 7]])
print(a + b)
print(a + 3)
print(a - b)
print(a - 2)
print(a * b)
print(a * 2)
print(a / b)
print(a / 2)
# 행렬의 형태가 다른 경우에는 불가
a = np.array([[1, 2, 3], [4, 5, 6]])
a = np.array([[2, 3], [5, 6]])
# 불가
# print(a + b)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[1, 2], [3, 4], [5, 6]])

print(np.dot(a, b))
