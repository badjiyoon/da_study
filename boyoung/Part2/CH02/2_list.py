# 자료형 리스트
a = [1, 3, 5, 5, 2, 4]

print(a)
print(type(a))

# 리스트 구조 / 인덱스 
# 리스트 내부에 리스트를 포함할 수도 있다
b = [1,[1,2],[1,[1,2]]]

print(b[0])
print(b[1][1])
print(b[2][1][0])

# 슬라이스
a = [1, [2, 3], [4,[5,6,7]]]
print(a[0])
print(a[-1])
print(a[1:])
print(a[2][1][1:])

# 치환
a = [1, [2,3], [4,[5,6,7]]]
# a[0]= 10 #바꿔야할 대상 입력
print(a)
a[2][1][1:] = [60, 70]
print(a)

# >> 리스트형태로 치환해야함
# a[2][1][1:] = 50 
# print(a)

print(a[2][1][1:])
a[2][1][1:] = [50] # 치환 해당하는 자료형 맞춰야함
print(a)

# 리스트 더하기
a = [1,2,3]
b = [4,5,6]
print(a+b)

# 리스트 곱하기
a = [1,2,3]
print(a * 2)

# append 함수: 해당 괄호안에 있는 뒤에 요소를 붙여줌
a = [1,2,3]
a.append(4)
print(a)

a = [1,2,3]
a.append([4,5])
print(a)

# pop : 맨 뒤에있는 요소 삭제
a = [1,2,3]
a.pop()
print(a)

#  구성요소 위치를 넣으면 해당 위치의 요소 삭제
a = [1,2,3]
a.pop(0) 
print(a)

# index : 리스트에 속해 있는 x의 위치값 반환
a = [1,2,3]
print(a.index(2))

# 튜플: 소괄호 사용
a = (1,2,3)
print(type(a))

# 인덱스
print(a[1])
# 슬라이스
print(a[:2])
# 치환 불가!(튜플 개인정보등 변경되면 안되는 변수에 사용)
# a[0] = 0

# 튜플 더하기 
a = (1,2,3)
b = (4,5,6)
c = a + b
print(c)

# 튜플 곱하기
a = (1,2,3)
b = a * 3
print(b)

# 자료형 셋
# 생성방법
# 리스트 셋 만들기
a = set([1,2,3]) 
print(a)
print(type(a))
# 튜플로 셋 만들기
a = set((1,2,3))
print(a)
print(type(a))

# 셋은 중복된 항목을 제거
a = set([1,1,2,3,3,4])
print(a)

# 셋은 순서가 없음
a = set([4,4,3,2,1,'a','b','a'])
print(a)
# print(a[0]) >> 에러발생(셋은 순서 없음, 인덱스 존재 X)

b = list(a) # 셋을 리스트로 변경
print(b)
print(b[0]) # 인덱스 가능함

# 합집합
a = set([1,2,3])
b = set([3,4,5])
c = a.union(b)
print(c)

# 교집합
a = set([1,2,3])
b = set([3,4,5])
c = a.intersection(b)
print(c)

# 차집합
a = set([1,2,3])
b = set([3,4,5])
c = a.difference(b)
print(c)
