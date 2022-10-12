# CH02_10. 제어문_while

'''
while: 반복하여 실행할 내용
'''

#문제: 1부터 10까지 출력하시오.
i=1
while i < 11:
    print(i)
    i=i+1

#문제: 1부터 9까지 2씩 증가하면서 출력하시오.
i=1
while i < 10:
    print(i)
    i+=2

#문제: 9부터 1까지 2씩 감소하면서 출력하시오.
i=9
while i > 0:
    print(i)
    i-=2

#문제: 1부터 10까지의 자연수의 합을 구하시오.
i=1
result=0
while i < 11:
    result += i
    i += 1
print(result)

#문제: 구구단을 출력하시오.
i=2
while i < 10:
    print(i)
    i += 1

i=2
while i < 10:
    j=1
    while j < 10:
        print(i*j)
        j += 1
    i += 1

i=2
while i < 10:
    j=1
    while j < 10:
        print('{} 곱하기 {} 은/는 {} 입니다.'.format(i, j, i*j))
        j += 1
    i += 1

