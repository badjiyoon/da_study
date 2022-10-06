# 1.변수 
# a = 100 b = 10000

# 100 + 10000
# 1000 * 10000
# 100 / 10000

a = 100
b = 10000

print(a)
print(b)

# 문제 4
# a 는 300 b는 400 이다.
# a + b, a - b, a * b, a / b
a = 300
b = 400

print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 변수명 작성시 주의사항 : 영문자, 숫자, _ 조합으로 구성

# 변수명 첫글자는 숫자 사용할 수 없음
# 1a= 1

# 대문자와 소문자는 구분되어 사용됨
a = 1
A = 10
print(a)
print(A)

# 변수명 사이는 띄어쓰기 불가 a b = 1


# 2. 데이터 형태
a = 1
b = 0.1
c = "hello"

print(type(a))
print(type(b))
print(type(c))

# 2-1 숫자형
# 정수 형태(int)
a = 1
print(type(a))

b = 0 
print(type(b))

c = -1
print(type(c))

# 실수 형태(float)
 
a = 1.1
print(type(a))
b = 0.0
print(type(b))
c = -1.1
print(type(c))
d = 1.
print(type(d))

# int와 float 의 연산결과는 int일까, float일까?
a = 1
b = 1.0

print(type(a+b)) # int 와 float 연산결과는 float

a = 10
b = 2

a + b
a - b
a * b
a / b

a = 10
b = 3

a // b # 몫 구하기 
a % b # 나머지 구하기
a ** b

# 할당연산
a = 1
#a = a + 1
a += 1
a -= 1
a *= 2
a /= 2

print(a)

# 2-2. 문자형
a = "hello"
a = 'hello'
a = 'hello world'
print(a)

# 인덱스 
# []를 통해 지정해 줄 수 있으며, 0 부터 시작
print(a[0]) #변수 a의 첫번째 글자 추출

print(a[-1]) #마지막 글자 추출

# 슬라이스
# 범위를 지정하여 원하는 부분만 추출
# a 전체 출력
print(a)
print(a[:])
# 변수 a의 6~8 번째 문자 출력
print(a[5:8])

# 뒤에서부터 5번째  마지막 문자 출력
print(a[-5:])

# 처음 - 뒤에서 부터 ~5번째 문자 출력
print(a[:-4])
# 문자열의 합
a = 'BO'
b = 'YOUNG'
print(a+b)
# 문자열의 곱셈
print(a*5)
# 문자열과 숫자열 함께 출력하기
# 방법1)
a =  'my number is '
b = 1111
print(a,b)
# 방법2)
# 숫자형을 문자형으로 변환하여 덧셈할수 있음
print(a +  str(b))
# 방법3)
# .format 문자열 내장함수 
print('My number is {}'.format(b))
# 2개 이상의 값 넣기
print('오늘은 {}월 {}일 입니다.'.format(10,3))

# 이름을 통한 값 넣기
print('오늘은 {month}월 {day}일 입니다. {month}월은 가정의 달 입니다.'.format(month = 5, day = 10))
ws = '오늘은 {month}월 {day}일 입니다. {month}월은 가정의 달 입니다.'.format(month = 5, day = 10)
print(ws)

# 문자열 분해
# hello 와 world 분해
# .split() 띄어쓰기 기준으로 분해
a = 'hello world'
a.split()

c = "i am a boy"
c.split()


# "."을 기준으로 분해하기
d = "i.am.a.boy"
d.split('.')