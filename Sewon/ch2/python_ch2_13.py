# CH02_13. 함수

print(1)

type(1)

'''
함수 생성 방법

def 함수명 (인풋):
    실행 내용
'''

#x를 입력값으로 넣으면 x+1을 출력하는 함수 생성
def function1(x):
    print(x+1)
function1(1)
function1(4)

#x를 입력값으로 넣으면 x+1을 반환하는 함수 생성
'''출력/반환 -> 출력은 그냥 보여주는 거고, 반환은 바꿔줌'''
def function2(x):
    result = x + 1
    return result
a=function2(10)
print(a)

#x를 입력값으로 넣으면 x+1, x+2를 반환하는 함수 생성
def function3(x):
    result1 = x+1
    result2 = x+2
    return result1, result2
a, b = function3(10)
print(a, b)

#여러 개의 입력값을 받고 싶을 때
#여러 개의 입력값을 받아 순서대로 출력

def function4(x):
    for i in x:
        print(i)

a=[1, 3, 5, 7, 4, 'd', 2, 60]
function4(a)

#args를 이용한 구현
'''변수 앞에 *를 붙이면 입력값들을 하나씩 다 넣는다'''
def function5(*some):
    for i in some:
        print(i)
function5(1, 3, 5, 7, 4, 'd', 2, 60)