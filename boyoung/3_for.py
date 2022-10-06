# 제어문
# 1) 반목문
# (1) for

# for 변수 in 반복 가능한 객체:
  # 반복하여 실행할 내용

# 반복가능한 객체: list, range

# 문제: 1부터 10까지 출력하시오
# list를 이용한 해결
a = [1,2,3,4,5,6,7,8,9,10]
for i in a:
    print(i)

# range를 이용한 해결
# range: 0 부터 해당숫자의 -1까지만을 가져옴
for i in range(11):
    print(i)

# 문제: 1부터 9까지 2씩 증가시키면서 출력하시오.
for i in range(1,10,2):
    print(i)

# 문제: 9부터 1까지 2씩 감소하면서 출력하시오
for i in range(9,0,-2):
    print(i)

# 문제: 1부터 10까지의 자연수의 합을 구하시오.
# result 초기값 지정 필요함!
result = 0
for i in range(1,11,1):
    result = result + i
print(result)


# 문제: 구구단을 출력하시오

for i in range(2,10,1):
    for j in range(1,10,1):
        print('{} 곱하기 {} 은/는 {} 입니다.'.format(i,j,i*j))
        

