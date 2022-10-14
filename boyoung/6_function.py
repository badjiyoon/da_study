# 6. 함수
print(1)
print(type(1))

# def 함수명(인풋 값): 실행내용

# x를 입력값으로 넣으면 x + 1을 출혁하는 함수 생성
def function(x):
    print(x+1)
function(1)

# x를 입력값으로 넣으면 x + 1을 반환하는 함수 생성
def function2(x):
    result = x + 1
    return result

a = function2(1)
print(a)

# x를 입력값으로 넣으면 x+1, x+2를 반환하는 함수 생성
def function3(x):
    result1 = x + 1
    result2 = x + 2
    return result1, result2

a, b = function3(1)
print(a)
print(b)

# 여러개의 입력값을 받고 싶을 때
# 여러개의 입력값을 받아 순서대로 출력
def function4(x):
    for i in x:
        print(i)

a = [1,3,5,7,4,'d',2,60]
function4(a)

# args를 이용한 구현
def function5(*args):
    for i in args:
        print(i)

function5(1,3,5,7,4,'d',2,60)