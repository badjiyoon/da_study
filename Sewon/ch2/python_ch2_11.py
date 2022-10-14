# CH02_11. 제어문_if

'''
if 문 작성법

if 조건1:
    조건1이 참 일때의 실행문
elif 조건2:
    조건2가 참 일때의 실행문
else:
    실행문
'''

#문제: x가 5보다 크면 True를 출력하시오.
x=6
if x > 5:
    print('True')

#문제: x가 5보다 크면 True, 아니면 False를 출력하시오.
x=4
if x > 5:
    print('True')
else:
    print('False')

#문제: x가 5보다 작으면 '5 미만', 5이면 '5', 나머지는 '5 이상'이라고 출력하시오.
x=4
if x < 5:
    print('5 미만')
elif x == 5:
    print('5')
else:
    print('5 미만')

#문제: x가 5보다 작으면 무시하고, 5이면 '5', 나머지는 '5 이상'이라고 출력하시오.
x=4
if x < 5: 
    pass
elif x == 5:
    print('5')
else:
    print('5 이상')

#문제: 점수(x)가 90점 이상이면 'A', 80점 이상이면 'B', 70점 이상이면 'C', 60점 이상이면 'D', 나머지는 'F'로 출력하시오.
x=95
if x >= 90:
    print('A')
elif x >= 80:
    print('B')
elif x >= 70:
    print('C')
elif x >= 60:
    print('D')
else:
    print('F')

#문제: A 리스트에는 1반 학생들의 수학 점수가 저장되어 있다.
#점수가 90점 이상이면 'A', 80점 이상이면 'B', 70점 이상이면 'C', 60점 이상이면 'D', 나머지는 'F'로 호출하시오.
A=[80, 95, 70, 55, 63]

for i in A:
    print(i)

for i in A:
    if i >= 90:
        print(i, 'A')
    elif i >= 80:
        print(i, 'B')
    elif i >= 70:
        print(i, 'C')
    elif i >= 60:
        print(i, 'D')
    else:
        print(i, 'F')