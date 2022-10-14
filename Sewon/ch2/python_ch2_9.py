# CH02_09. 제어문_for

'''
for 변수 in 반복 가능한 객체:
반복하여 실행할 내용

반복 가능한 객체: list, range
'''

#문제: 1부터 10까지 출력하시오.

a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in a: 
    print(i) 

for i in range(11): #range는 0부터 n-1까지 출력
    print(i)

for i in range(1, 10, 2): #1부터 9까지 2씩 증가하면서 출력
    print(i)

for i in range(9, 0, -2): #9부터 1까지 2씩 감소하면서 출력
    print(i)

for i in range(1, 11, 1): #1부터 10까지 자연수의 합 출력
    print(i)

result=0
for i in range(1, 11, 1): 
    result = result + i
print(result)

for i in range(2, 10, 1):
    print(i)

for i in range(2, 10, 1):
    for i in range(1, 10, 1):
        print(i * i)

for i in range(2, 10, 1):
    for j in range(1, 10, 1):
        print(i * j)

for i in range(2, 10, 1):
    for j in range(1, 10, 1):
        print('{} 곱하기 {} 은/는 {} 입니다.'.format(i, j, i * i))
