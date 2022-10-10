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

