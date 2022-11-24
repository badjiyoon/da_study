# CH02_04. 자료형_문자형

a='hello'
b="hello"

#인덱스
a='hello world'
print(a[0]) #변수 a의 첫 번째 문자 출력
print(a[-1]) #변수 a의 마지막 문자 출력

#슬라이스
a='hellow world'
print(a)
print(a[:]) # : (콜론)은 전체를 의미
print(a[5:8]) #변수 a의 6~8번째 문자 출력, [5:8]=5 이상 8 미만
print(a[:5]) #처음~5번째 문자 출력
print(a[-5:]) #뒤에서부터 5번째~마지막 문자 출력
print(a[:-4]) #뒤에서부터 -5번째 문자 출력

#문자열의 합
a='hello'
b='world'
print(a+b)

#문자열의 곱
a='hello'
print(a*3)

#문자형과 숫자형의 혼용

#방법1
a='My number is '
b=1
print(a, b)

#방법2
a='My number is '
b=1
print(a+b)
'''문자형과 숫자형을 같이 연산할 수 없음'''
print(a+str(b))
'''문자형과 숫자형을 더해주기 위해서는 숫자형을 문자형으로 바꿔야 함'''

#방법3
print('My number is {}'.format(b))

#두 개 이상의 값 넣기
print('오늘은 {}월 {}일 입니다.'.format(5, 10))
print('오늘은 {month}월 {day}일 입니다. {month}월은 가정의 달입니다.'.format(month=5, day=10))
a='오늘은 {month}월 {day}일 입니다. {month}월은 가정의 달입니다.'.format(month=5, day=10)
print(a)

#문자열 분해
a='hello world'
a.split()

b='I am a boy'
b.split()

b='I.am.a.boy'
b.split()
b.split('.')