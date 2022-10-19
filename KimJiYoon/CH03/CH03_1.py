# apply, map을 활용한 데이터 변환
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
# a가 2보다 작으면 '2미만', 4보다 작으면 '4미만', 4보다 크면 '4이상'이 저징된 b칼럼을 추가하시오
print(df)
df['b'] = 0
print(df)
a = df[df['a'] < 2]
print(a)
df['b'][a.index] = '2 미만'
print(df)
a = df[(df['a'] >= 2) & (df['a'] < 4)]
df['b'][a.index] = '4 미만'
pd.set_option('mode.chained_assignment', None)
print(df)
a = df[df['a'] >= 4]
df['b'][a.index] = '4 이상'
print(df)


# 함수 + apply를 이용한 해결
def case_function(x):
    if x < 2:
        return '2 미만'
    elif x < 4:
        return '4 미만'
    else:
        return '4 이상'


# apply와 사용자 정의함수로 처리
df['c'] = df['a'].apply(case_function)
print(df)


# a가 1이면 'one', 2이면 'two', 3이면 'three', 4이면 'four', 5이면 'five'를 출력하는 컬럼 D를 만드시오.
def engFunction(x):
    if x == 1:
        return 'one'
    elif x == 2:
        return 'two'
    elif x == 3:
        return 'three'
    elif x == 4:
        return 'four'
    elif x == 5:
        return 'five'


df['d'] = df['a'].apply(engFunction)
print(df)

a = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
df['e'] = df['a'].map(a)
print(df)

# 데이터 프레임 결합
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 12, 13], 'C': [21, 22, 23]})
df2 = pd.DataFrame({'A': [4, 5, 6], 'B': [14, 15, 16], 'C': [24, 25, 26]})

# 새로 결합
print(pd.concat([df1, df2]))
# 거꾸로 새로
print(pd.concat([df2, df1]))
# index 초기화를 위해서는 ignore_index = True
print(pd.concat([df2, df1], ignore_index=True))

df2 = pd.DataFrame({'B': [14, 15, 16], 'A': [4, 5, 6], 'C': [24, 25, 26]})
print(pd.concat([df1, df2]))

# 서로 다른 필드로 구성되어있는 데이터 플레임의 결합
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [11, 12, 13], 'C': [21, 22, 23], 'D': [31, 32, 33]})
df2 = pd.DataFrame({'A': [3, 4, 5], 'B': [14, 15, 16], 'C': [24, 25, 26], 'E': [41, 42, 43]})
print(pd.concat([df1, df2], join='outer'))
print(pd.concat([df1, df2], join='inner'))

# 좌우 결합
df2 = pd.DataFrame({'E': [3, 4, 5], 'F': [13, 14, 15], 'G': [24, 25, 26], 'H': [41, 42, 43]})
print(df1)
print(df2)
print(pd.concat([df1, df2], axis=1))
# 다음의 두 데이터 프레임을 결합하시오
df1 = pd.DataFrame({'ID': [1, 2, 3], '성별': ['F', 'M', 'F'], '나이': [20, 30, 40]})
df2 = pd.DataFrame({'ID': [1, 2, 3], '키': [160.5, 170.3, 180.1], '몸무게': [45.1, 50.3, 72.1]})
print(df1)
print(df2)
print(pd.concat([df1, df2], axis=1))
df1 = pd.DataFrame({'ID': [1, 2, 3, 4, 5], '성별': ['F', 'M', 'F', 'M', 'F'], '나이': [20, 30, 40, 25, 42]})
df2 = pd.DataFrame({'ID': [3, 4, 5, 6, 7], '키': [160.5, 170.3, 180.1, 142.3, 153.7], '몸무게': [45.1, 50.3, 72.1, 38, 42]})
print(df1)
print(df2)
print(pd.concat([df1, df2], axis=1))
# LEFT JOIN, INNER JOIN
# 성별과 나이가 확인 된 유저들을 대상으로 키와 몸무게의 정보를 결합하시오
print(pd.merge(df1, df2, how='left', on='ID'))
# 키와 몸무게가 확인된 유저들을 대상으로 성별과 나이의 정보를 결합하시오
print(pd.merge(df2, df1, how='left', on='ID'))
print(pd.merge(df1, df2, how='right', on='ID'))
# 키, 몸무게, 성별, 나이 정보가 모두 확인 된 유저들의 정보를 출력하시오
print(pd.merge(df1, df2, how='inner', on='ID'))
# 모든 유저들의 정보를 출력하시오 (합집합)
print(pd.merge(df1, df2, how='outer', on='ID'))
# 모든 유저들의 정보를 출력하시오
df1 = pd.DataFrame({'USER_ID': [1, 2, 3, 4, 5], '성별': ['F', 'M', 'F', 'M', 'F'], '나이': [20, 30, 40, 25, 42]})
print(pd.merge(df1, df2, how='outer', left_on='USER_ID', right_on='ID'))
df1 = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    '가입일': ['2021-01-02', '2021-01-04', '2021-01-10', '2021-02-10', '2021-02-24'],
    '성별': ['F', 'M', 'F', 'M', 'M']
})

df2 = pd.DataFrame({
    '구매순서': [1, 2, 3, 4, 5],
    'ID': [1, 1, 2, 4, 1],
    '구매월': [1, 1, 2, 2, 3],
    '금액': [1000, 1500, 2000, 3000, 4000]
})

print(df1)
print(df2)
print(pd.merge(df1, df2, how='left', on='ID'))

# 그룹화
# df1은 회원의 정보를 저장한 데이터 프레임이며, df2는 각 회원의 구매 내역을 저장한 데이터 프레임이다.
# 각 회원의 누적 금액을 회원ID별로 구하시오
print(df2)
print(df2.groupby(by=['ID'])['금액'].sum())
print(type(df2.groupby(by=['ID'])['금액'].sum()))
s2 = df2.groupby(by=['ID'])['금액'].sum()
print(pd.merge(df1, s2, how='left', on='ID'))
# 월별 누적금액 구하기
print(df2.groupby(by=['ID', '구매월'])['금액'].sum())
s2 = df2.groupby(by=['ID', '구매월'])['금액'].sum()
print(s2.index)
# 구매월 누락
print(pd.merge(df1, s2, how='left', on='ID'))
df3 = pd.DataFrame(s2)
print(df3)
print(df3.index)
print(pd.merge(df1, s2, how='left', on='ID'))
print(df2.groupby(by=['ID', '구매월'], as_index=False)['금액'].sum())
print(type(df2.groupby(by=['ID', '구매월'], as_index=False)['금액'].sum()))
df3 = df2.groupby(by=['ID', '구매월'], as_index=False)['금액'].sum()
print(pd.merge(df1, df3, how='left', on='ID'))

df = pd.DataFrame({
    '구매순서': [1, 2, 3, 4, 5],
    'ID': [1, 1, 2, 4, 1],
    '구매월': [1, 1, 2, 2, 3],
    '금액': [1000, 1500, 2000, 3000, 4000],
    '수수료': [100, 150, 200, 300, 400]
})
# 누적금액과 누적 구매 횟수 회원ID 별로 구하시오
print(df)
print(df.groupby(by=['ID'])['금액'].agg([sum, len]))
print(df.groupby(by=['ID'], as_index=False)['금액'].agg([sum, len]))
df2 = df.groupby(by=['ID'])['금액'].agg([sum, len])
df2.reset_index(inplace=True)
print(df2)
# 각 회원의 최대 사용금액 / 최소 사용 금액과 최저 수수료의 값을 구하시오
print(df)
print(df.groupby(by=['ID']).agg({'금액': [max, min], '수수료': [min]}))
df2 = df.groupby(by=['ID']).agg({'금액': [max, min], '수수료': [min]})
df2.reset_index()
print(df2.columns)
print(df2.columns.values)
df2.columns = ['_'.join(col) for col in df2.columns.values]
print(df2)
print(df.reset_index())
