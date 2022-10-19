import pandas as pd
import random
import numpy as np

# 문제 : 다음 데이터 프레임은 A 서비스의 월별 탈퇴 회원수를 가입 월별로 분류해 놓은것이다.
# 이 데이터 프레임을 이용하여 피벗 테이블을 만드시오
df = pd.DataFrame({
    '가입월': [1, 1, 1, 2, 2, 3],
    '탈퇴월': [1, 2, 3, 2, 3, 3],
    '탈퇴회원수': [101, 52, 30, 120, 60, 130]
})

print('탈퇴회원수 데이터 : ', df)
pivot = pd.pivot_table(df, values='탈퇴회원수', index=['가입월'], columns=['탈퇴월'])
print('pivot table 타입 : ', type(pivot))
print(pivot)
# NAN값 0으로 채우기
print(pd.pivot_table(df, values='탈퇴회원수', index=['가입월'], columns=['탈퇴월'], fill_value=0))
# 다음 데이터 프레임은 어느 과일 매장의 판매내역이다. 각 상품 항목 별, 크기 별로 판매 개수와 판매 금액의 합을 구하시오.
print(random.randint(1, 3))
# 변수 처리
a = []
b = []

for i in range(100):
    a.append(random.randint(1, 3))
    b.append(random.randint(1, 3))

df = pd.DataFrame({
    '품목': a,
    '크기': b
})

print(df)

df['금액'] = df['품목'] * df['크기'] * 500
df['수수료'] = df['금액'] * 0.1

print(df)

fruit_name = {
    1: '토마토',
    2: '바나나',
    3: '사과'
}

fruit_size = {
    1: '소',
    2: '중',
    3: '대'
}

df['품목'] = df['품목'].map(fruit_name)
df['크기'] = df['크기'].map(fruit_size)
print(df)
print(pd.pivot_table(df, values='금액', index=['품목'], columns=['크기'], aggfunc=('count', 'sum')))
# 다음 데이터 프레임은, 어느 과일 매장의 판매내역이다. 각 상품 항목 별 크기 별로 판매 개수와 판매 금액 / 수수료의 합을 구하시오
print(df)
print(pd.pivot_table(df, index=['품목'], columns=['크기'], aggfunc={'금액': ['count', 'sum'], '수수료': 'sum'}))
