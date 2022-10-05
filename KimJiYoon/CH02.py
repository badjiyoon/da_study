# 1. 파이썬의 특징
# * 세미콜론(;)으로 끝을 맺지 않아도 됩니다.
import re
# 외부라이브러 사용법
import pandas
import pandas as pd

print('Hello World')
# * 세미콜런(;)으로 끝을 맺어도 됨.
print('Hello World');

# * 들여쓰기에 주의가 필요합니다.
print(1 + 1)

# * 주석은 #(한 줄), 또는 ". ""로 만들 수 있습니다.
'''
이렇게
여러줄을
주석처리 할 수 있습니다.
'''

"""
    쌍 따옴표로도
할 수 있습니다.
"""

# 2. 변수# 
# * 문제 1
# a = 100, b는 10000이다.
# a + b를 구하시오
# 변수 생성하는 법1
# 자주 사용하는 숫자들을 변수에 저장하여 재 사용한다면??
a = 100
b = 10000
a, b = 100, 10000
print(a)
print(b)
print(a + b)
# a * b를 구하시오
print(a * b)
# a / b를 구하시오
print(a / b)

# 문제 4
# - 문제 4
# a 는 300, b 는 400 이다.
# a + b, a - b, a * b, a / b 를 출력하시오.
a, b = 300, 400
print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 변수명 작성시 주의사항
# 영문자, 숫자, _의 조합으로 구성되어야함.
a = 1
a1 = 1
_ = 1

# 변수명의 첫 글자는 숫자로 사용할 수 없음
# 1a = 1
# 대문자와 소문자는 구분되어 사용함
a = 1
A = 10
print(a)
print(A)

# 변수명 사이에 띄워쓰기는 불가
# a b = 1

# 3. 자료형
# * 저장되는 데이터의 형태
a = 1
b = 0.1
c = 'hello'

print(type(a))
print(type(b))
print(type(c))

# 1) 숫자형
# int : 정수의 형태
a = 1
print(type(a))
b = 0
print(type(b))
c = -1
print(type(c))

# 2) 실수형
# float : 실수의 형태
a = 1.1
print(type(a))
b = 0.0
print(type(b))
c = -1.1
print(type(c))
d = 1.
print(type(d))

# int와 float의 연산결과느 int일까, float일까?
a = 1
b = 1.0
print(type(a))
print(type(b))
print(type(a+b))

# 연산
# 더하기
a = 10
b = 2
print(a + b)
# 빼기
print(a - b)
# 곱하기
print(a * b)
# 나누기
print(a / b)
# 몫
a = 10
b = 3
print(a // b)
# 나머지
print(a % b)
# 제곱
a ** b
# 할당연산
# 변수 a는 1이다 해당 변수 a 에 1을 더하라
a = 1
a = a + 1
print(a)
# 빼기
a = 1
a -= 1
print(a)
# 홉하기
a = 2
a += 2
print(a)
# 나누기
a = 1
a /= 2
print(a)

#문자형
# * 문자형을 만들기 위해서는 문자를 따옴표(') 쌍 따옴표(")로 감싸준다.
a = 'hello'
a = "hello"
# 인덱스
# 인덱스는 []를 통해 지정해줄 수 있으며, 0부터 시작합니다.
a = 'hello world'
print(a[0])
print(a[-1])
# 슬라이스
# * 범위를 지정하여 원하는 부분만을 얻을 수 있습니다.
# 변수전체 출력
print(a)
print(a[:])

# 변수 a의 6 ~ 8번째 문사를 출력하시오.
print(a[5:8])

#변수 a의 처음 ~ 5번째 문자를 출력하시오
print(a[:5])
#변수 a의 5번쨰 ~ 마지막 문자를 출력하시오
print(a[-5:])
#변수 a의 처음 ~ 뒤에서부터 -5번쨰 문자를 출력하시오
print(a[:-4])
# * 문자열의 합
a = 'hellow'
b = 'world'
print(a + b)

a = 'hellow'
print(a * 3)
# 문자형과 숫자형의 혼용
# * 문제 : 변수를 넣어 'My number is (변수)'를 출력하시오
# 방법 1
a = 'My number is '
b = 1
print(a, b)
# 방법 2 에러
# print(a + b)
# 방법3
print(a + str(b))
print('My number is {}'.format(b))
# 두 개 이상의 값 넣기
print('오늘은 {}월 {}일 입니다.'.format(5, 10))
#이름을 통한 값 넣기
print('오늘은 {month}월 {day}일 입니다. {month}월은 가정의 달입니다.'.format(month = 5, day = 10))
a = '오늘은 {month}월 {day}일 입니다. {month}월은 가정의 달입니다.'.format(month = 5, day = 10)
print(a)
# 문자열 분해
a = 'hello world'
print(a.split())

a  = 'i am a boy'
print(a.split())

a = 'i.am.a.boy'
print(a.split('.'))

# 리스트
a1 = 1
a2 = 3
a3 = 5
a4 = 5
a5 = 2
a6 = 4

a = [1, 3, 5, 5, 2, 4]
print(a)
print(type(a))
# 리스트의 구조 / 인덱스
# * 리스트 내부에 리스트를 포함할 수도 있다.
b = [1, [1, 2], [1, [1, 2]]]
print(b)
print(b[0])
print(b[1][1])
print(b[2][1][0])

a = [1, [2, 3], [4, [5, 6, 7]]]
print(a[0])
print(a[-1])
print(a[1:])
print(a[2][1][1:])
# 치환
a[0] = 10
print(a)

a[2][1][1:] = [60, 70]
print(a)

print(a[2][1][1:])
a[2][1][1:] = [50]
print(a)

# 리스트 더하기
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)
a = [1, 2, 3]
print(a * 2)
# appen
a = [1, 2, 3]
a.append(4)
print(a)
a = [1, 2, 3]
a.append([4, 5])
print(a)
# pop
a = [1, 2, 3]
a.pop()
print(a)
# index
# * 리스트에 속해 있는 x의 위치 값 반환
a = [1, 2, 3]
print(a.index(2))

# 4) 튜플
a = (1, 2, 3)
print(type(a))
print(a[1])

# 슬라이스
a = (1, 2, 3)
print(a[:2])
# 치환하기
a = (1, 2, 3)
# 튜플은 변환 불가 에러
# a[2] = 0
# 튜플 더하기
a = (1, 2, 3)
b = (4, 5, 6)
c = a + b
print(c)
# 튜플 곱하기
a = (1, 2, 3)
b = a * 2
print(b)

# 딕셔너리
# * 생성방법 : 딕셔너리느 key와 value로 구성
a = {'사자' : 'lion', '호랑이' : 'tiger', '용' : 'dragon'}
print(a)
print(type(a))
# 하나의 Key에 여러개의 Value로도 구성이 가능
a = {'car' : ['bus', 'truck', 'taxi'], 'train' : 'ktx'}
print(a)
# key 얻기
key = a.keys()
print(key)
print(type(key))
# 에러 키의 순서로는 못꺼냄
# print(key[0])
key2 = list(key)
print(type(key2))
print(key2[0])
# value 얻기
value = a.values()
print(value)
print(type(value))
# 에러
# print(value[0])
value2 = list(value)
print(type(value2))
print(value2[0])
a = {'car' : ['bus', 'truck', 'taxi'], 'train' : 'ktx'}
# 요소 추가하기
a['plane'] = 'jet'
print(a)
# 요소 삭제하기
a = {'car' : ['bus', 'truck', 'taxi'], 'train' : 'ktx'}
del a['car']
print(a)

# Set
# 리스트로 처리
a = set([1, 2, 3])
print(a)
print(type(a))

# 튜ㄹㅗ 처리
a = set((1, 2, 3))
print(a)
print(type(a))

# 중복을 제거한다.
# Set의 경우 중복된 원소를 제거한 값을 리턴한다.
a = set([1, 1, 2, 3, 3, 4])
print(a)

# 순서가 없다
a = set([4, 4, 3, 2, 1, 'a', 'b', 'a'])
print(a)
# 오류가 난다. 셋은 리스트가 아님
# print(a[0])
# 리스트 형변화을 이용해 처리
b = list(a)
print(b)
print(b[0])

# 합집합
a = set([1, 2, 3])
b = set([3, 4, 5])
c = a.union(b)
print(c)

# 교집합
c = a.intersection(b)
print(c)

# 차집합
c = a.difference(b)
print(c)

# 제어문 for
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# for의 가장 기본형
# 문제 : 1 ~ 10까지 출력하라
for i in a :
    print(i)

# range 함수를 이용한 처리 1 ~ 10까지 처리를 위한 순서 처리
for i in range(1, 11) :
    print(i)

# 문제 : 1부터 9까지 2씩 증가하면서 출력하시오.
for i in range(1, 10, 2) :
    print(i)

# 문제 : 9부터 1까지 2씩 감소하면서 출력하시오
for i in range(9, 0, -2) :
    print(i)

# 문제 : 1부터 10까지 자연수의 합을 구하시오
result = 0
for i in range(1, 11, 1) :
    result += i
    print(result)

# 문제 : 구구단을 출력하시오
for i in range(1, 10) :
    for j in range(1, 10) :
        print('{0} * {1} = {2}'.format(i, j, i*j))

# while
# 문제 : 1부터 10까지 출력하시오
i = 1
while (i < 11) : 
    print(i)
    i += 1

# 문제 : 1부터 9까지 2씩 증가하면서 출력하시오
i = 1
while (i < 10) :
    print(i)
    i += 2

# 문제 : 9부터 1까지 2씩 감소하면서 출력하시오
i = 9
while (i > 0) :
    print(i)
    i -= 2

# 문제 : 1부터 10까지 자연수의 합을 구하시오
i = 1
result = 0
while (i < 11) :
    result += i
    print('result = ', result)
    i += 1

# 문제 : 구구단을 출력하시오
i = 1
while (i < 10) :
    j = 1
    while (j < 10) :
        print('{0} * {1} = {2}'.format(i, j, i*j))
        j += 1
    i += 1

# if문
# 문제 : x가 5보다 크면 True를 출력하시오
x = 6
if x > 5 :
    print(True)

# 문제 : x가 5보다 크면 True 아니면 False를 출력하시오
x = 4
if x > 5 :
    print(True)
else :
    print(False)

# 문제 : x가 5보다 작으면 '5미만' 5이면 '5' 나머지는 '5이상' 이라고 출력하시오
x = 4
if (x < 5) :
    print('5미만')
elif (x == 5) :
    print('5')
else :
    print('5이상')

# 문제 : 점수(x)가 90점 이상이면 'A', 80점 이상이면 'B', 70점 이상이면 'C' 60점 이상이면 'D'
# 나머지는 'F'로 출력하시오
x = 95
if x >= 90 :
    print('A')
elif x >= 80 :
    print('B')
elif x >= 70 :
    print('C')
elif x >= 60:
    print('D')
else : 
    print('F')

# 문제 : A리스트에느 1반 학생들의 수학 점수가 저장되어 있다. 점수가 90점이상이면 'A' 80점 이상이면 'B'
# 70점 이상이면 'C' 60점이상이면 'D' 나머지는 'F'로 출력하시오

A = [80, 95, 70, 55, 63]
for i in A :
    print(i)

for i in A :
    if i >= 90 :
        print(i, 'A')
    elif i >= 80 :
        print(i, 'B')
    elif i >= 70 :
        print(i, 'C')
    elif i >= 60:
        print(i, 'D')
    else : 
        print(i, 'F')

# List Comprehension
for i in range(1, 11, 1):
    a.append(i)
print(a)

a = [i for i in range(1, 11, 1)]
print(a)

a = []
for i in range(1, 10) :
    a.append(i ** 2)
print(a)

a = [i ** 2 for i in range(1, 10)]
print(a)

a = [90, 39, 48, 70, 82, 100]

result = [i for i in a]
print(result)

result = [i for i in a if i >= 80]
print(result)

# 함수
def function1(x) : 
    print(x + 1)

def function2(x) :
    result = x + 1
    return result

def function3(x) :
    result1 = x + 1
    result2 = x + 2
    return result1, result2

def function4(x):
    for i in x :
        print(i)

# args를 이용한 구현
def function5(*args) :
    for i in args :
        print(i)

function1(1)
a = function2(1)
print(a)

a, b = function3(1)
print(a)
print(b)

a = [1, 3, 5, 7, 4, 'd', 2, 60]
function4(a)
function5(1, 3, 5, 7, 4, 'd', 2, 60)

df_sample1 = pandas.DataFrame({'a' : [1, 2, 3], 'b' : [4, 5, 6]})
df_sample2 = pd.DataFrame({'a' : [1, 2, 3], 'b' : [4, 5, 6]})

print(df_sample1.head())
print(df_sample2.head())