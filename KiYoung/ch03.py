import pandas as pd
import numpy as np
import copy

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(type(df))
print(df)

dummy = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
df2 = pd.DataFrame(dummy)
print(df2)

# List를 이용한 데이터 프레임 생성
a = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
df3 = pd.DataFrame(a)
print(df3)

# 컬럼 명칭 등록
df3.columns = ['a', 'b', 'c']
print(df3)

# 다양한 Type을 가지고 가능한지 판단한다.
df4 = pd.DataFrame({'company': ['abc', '회사', 123], '직원수': [400, 10, 6]})
print(df4)

# NaN을 처리하기 위해서는 numpy와 과련하여 작업하기 위해 사용함.
df5 = pd.DataFrame({'company': ['abc', '회사', 123], '직원수': [400, np.NaN, 6]})
print(df5)

# 컬럼명 추출 / 변경
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)

# 컬럼 보기
print(df.columns)
print(df.columns[1])

# 문제 : 컬럼명인 a, b, c를 d, e, f로 바꾸어라
df.columns = ['d', 'e', 'f']
print(df)

df.columns = ['디', 'e', '에프']
print(df)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
df.columns = ['d', 'e', 'f']
print(df)

print(df.rename(columns={'d': '디', 'f': '에프'}))
print(df)

# inplace 옵션을 주어야 헤더 치환됨
df.rename(columns={'d': '디', 'f': '에프'}, inplace=True)
print(df)

# copy를 이용한 데이터 복사
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
df.columns = ['d', 'e', 'f']
print(df)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
df.rename(columns={'a': '에이'}, inplace=True)
print(df)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
# 얕은 복사
# df2 = df
# 깊은 복사
df2 = copy.deepcopy(df)
# df2 = df.copy()

print(df)
print(df2)
df.columns = ['d', 'e', 'f']
print(df)
print(df2)

# 시리즈
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
print(df['a'])
print(type(df['a']))

# 시리즈 생성 방법
# 인덱스 와 짝으로 생성됨
a = pd.Series([1, 2, 3, 1, 2, 3])
print(a)

a = pd.Series([1, 2, 3, 1, 2, 3], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(a)
print(a['e'])

# 유일한 값 찾기
df = pd.DataFrame({'a': [1, 2, 3, 1, 2, 3], 'b': [4, 5, 6, 6, 7, 8], 'c': [7, 8, 9, 10, 11, 12]})
a = df['a']
print(a)
print(type(a))
print(a.unique())
print(a.unique()[2])

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
# 에러 두개의 시리즈로는 안만들어짐
# print(df['a', 'b'])

# loc과 iloc을 이용한 원하는 위치의 데이터 추출
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]})
print(df)
# 에러
# df['a', 'b']
print(df[['a', 'b']])
print(type(df[['a', 'b']]))

# print(df[0])

# 인엑스에서 뽑기
print(df.loc[0])
# 슬라이스도 가능
print(df.loc[2:4])
# 인덱스가문자로 이루어진 데이터 프레임 생성
index = ['a', 'b', 'd', 'c', 'e', 'f', 'g', 'g', 'h', 'i']
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]},
                  index=index)
print(df)
# 불가 인덱스가 바껴서
# print(df.loc[0])
print(df.loc['g'])
print(df.loc['c':])
print(df.loc[['g', 'i'], ['a', 'c']])

# 처음부터 5번쨰까지의 데이터와 첫번쨰 열과 세번쨰 열의 데이터를 추출하시오
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]})
print(df)

print(df.iloc[:5, [0, 2]])

index1 = ['a', 'b', 'd', 'c', 'e', 'f', 'g', 'g', 'h', 'i']
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]},
                  index=index1)
print(df)
print(df.iloc[:5, [0, 2]])

# 조건에 맞는 데이터 추출
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]})
print(df)

print(df[['a', 'c']])
# 3보다 큰 데이터를 출력하라
print(df[df['a'] >= 3])
# a가 3이상인 데이터 중 a,열만 출력하시오
print(df[df['a'] >= 3][['a', 'c']])
# a가 3이상이고 b가 16미만인 데이터를 출력하시오
print(df[(df['a'] >= 3) & (df['b'] < 16)])
a = (df['a'] >= 3) & (df['b'] < 16)
print(a)
type(a)
# 같은 방식 시리즈로 DataFrame에서 뽑아온다.
print(df[a])
# a가 3이하 이거나 7이상인 데이터를 출력하시오.
print(df[(df['a'] <= 3) | (df['a'] >= 7)])
# a가 3이상이고 b가 16미만이거나 c가 30인 데이터
print(df[(df['a'] >= 3) & ((df['b'] < 16) | (df['c'] == 30))])

# 정렬
df = pd.DataFrame({'a': [2, 3, 2, 7, 4], 'b': [2, 1, 3, 5, 3], 'c': [1, 1, 2, 3, 5]})
print(df)
print(df.sort_index())
print(df.sort_index(ascending=False))
# 저장 시 inplace True
df.sort_index(ascending=False, inplace=True)
print(df)
# 인덱스만 초기화
print(df.reset_index())
# 인덱스 삭제
print(df.reset_index(drop=True))
# 인덱스 삭제 및 적용
df.reset_index(drop=True, inplace=True)
print(df)
# 값 기준 정렬
df = pd.DataFrame({'a': [2, 3, 2, 7, 4], 'b': [2, 1, 3, 5, 3], 'c': [1, 1, 2, 3, 5]})
print(df)

# a열 기준으로 오름차순 정렬을하시오
print(df.sort_values(by=['a']))
df.sort_values(by=['a'], inplace=True)
print(df)
print(df.sort_values(by=['a'], ascending=False))
# a,b 열 기준으로 오름차순 정렬하시오
print(df.sort_values(by=['a', 'b']))
# a열 기준으로 오름차순 정렬한 이후, b열 기준으로 내림차순 정렬하시오.
print(df.sort_values(by=['a', 'b'], ascending=[True, False]))
df.reset_index(drop=True, inplace=True)

# 결측값 처리
df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})
print(df)
# 결측 유무 확인
print(df.isnull())
# 결측값 개수 확인
print(df.isnull().sum())
# 결측값이 포함된 행 지우기
print(df.dropna())
df.dropna(inplace=True)
print(df)
# 결측값이 포함된 열 지우기
df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})
print(df.dropna(axis=1))
df.dropna(axis=1, inplace=True)
print(df)
# 결측값을 다른값으로 대체하기
df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})
print(df)
df.fillna(0, inplace=True)
print(df)
# 열이나 뒤의 숫자로 바꾸기
df = pd.DataFrame({'a': [1, 1, 3, 4, np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 7, 6, 4]})
print(df)
# 뒤의 값으로 채우기
print(df.fillna(method='bfill'))
print(df.fillna(method='ffill'))
# limit 설정
print(df.fillna(method='ffill', limit=1))
# 데이터 프레임에 존재하는 결측값들을 뒤의 값으로 대체한 이후 앞의 값으로 대체하시오
df = pd.DataFrame({'a': [1, 1, 3, 4, np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 7, 6, 4]})
print(df)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)
print(df)
# 평균으로 대체
df = pd.DataFrame({'a': [1, 1, 3, 4, np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 1, 1, 4]})
print(df)
df.mean()['a']
print(df.fillna(df.mean()['a']))
df.mean()
print(df.fillna(df.mean()))
print(df)
print(df.fillna(df.mean()[['b', 'c']]))

# 타입 변환
df = pd.DataFrame({
    '판매일': ['5/11/21', '5/12/21', '5/13/21', '5/14/21', '5/15/21'],
    '판매량': ['10', '15', '20', '25', '30'],
    '방문자수': ['10', '-', '17', '23', '25'],
    '기온': ['24.1', '24.3', '24.8', '25', '25.4']
})
print(df)
# 타입 확인
print(df.dtypes)
# 1더하는게 안됨
# df['판매량 보정'] = df['판매량'] + 1
df.astype({'판매량': 'int'})
print(df.dtypes)
df = df.astype({'판매량': 'int'})
print(df.dtypes)
df['판매량 보정'] = df['판매량'] + 1
print(df)
# 방문자수를 숫자 타입으로 변환하시오
# 옵션 써야함
# df.astype({'방문자수': 'int'})
df['방문자수'] = pd.to_numeric(df['방문자수'], errors='coerce')
print(df.dtypes)
print(df)
# 에러남 nan 때문에
# df = df.astype({'방문자수': 'int'})
df.fillna(0, inplace=True)
df = df.astype({'방문자수': 'int'})
print(df)
print(df.dtypes)
df['판매일'] = pd.to_datetime(df['판매일'], format='%m/%d/%y')
print(df)
print(df.dtypes)

# 타입 변환
df = pd.DataFrame({
    'a': [1, 1, 3, 4],
    'b': [2, 3, 2, 3],
    'c': [3, 4, 7, 4]
})
 
df['af'] = [1, 3, 6, 4, 8]
print(df)

df['ce'] = [1, 2, 1, 1, 1]
print(df)
df['e'] = 1
print(df)
print(df.dtypes)
df['f'] = df['a'] + df['b'] - df['c']
print(df)
df.drop(['d', 'e', 'f'], axis=1)
print(df)
df.drop(['d', 'e', 'f'], axis=1, inplace=True)
print(df)
df.append({'a': 6, 'b': 7, 'c': 8}, ignore_index=True)
print(df)
df = df.append({'a': 6, 'b': 7, 'c': 8}, ignore_index=True)
print(df)

df.loc[6] = [7, 8, 9]
print(df)

print(df)
df.drop(0)
print(df)
df = df.drop([0, 1])
print(df)
df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, 2, 3, 4], 'c': [3, 4, 7, 6, 4]})
print(df)
print(df.drop([i for i in range(4)]))
print(df)
print(df.drop(df.index[:4]))
