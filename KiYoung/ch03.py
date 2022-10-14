import pandas as pd
import numpy as np
import copy

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(type(df))
print(df)

dummy = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
df2 = pd.DataFrame(dummy)
print(df2)
sa = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
df3 = pd.DataFrame(sa)
print(df3)

df3.columns = ['aa', 'bv', 'cc']
print(df3)

df4 = pd.DataFrame({'company': ['abc', '회사', 123], '직원수': [400, 10, 6]})
print(df4)
df5 = pd.DataFrame({'company': ['abc', '회사', 123], '직원수': [400, np.NaN, 6]})
print(df5)
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)

print(df.columns)
print(df.columns[1])

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
df.rename(columns={'d': '디', 'f': '에프'}, inplace=True)
print(df)
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
df.columns = ['d', 'e', 'f']
print(df)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
df.rename(columns={'a': '에이'}, inplace=True)
print(df)

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df2 = copy.deepcopy(df)
print(df)
print(df2)
df.columns = ['d', 'e', 'f']
print(df)
print(df2)
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
print(df['a'])
print(type(df['a']))
a = pd.Series([1, 2, 3, 1, 2, 3])
print(a)

a = pd.Series([1, 2, 3, 1, 2, 3], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(a)
print(a['e'])
df = pd.DataFrame({'a': [1, 2, 3, 1, 2, 3], 'b': [4, 5, 6, 6, 7, 8], 'c': [7, 8, 9, 10, 11, 12]})
a = df['a']
print(a)
print(type(a))
print(a.unique())
print(a.unique()[2])

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df)
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]})

print(df.loc[0])

print(df.loc[2:4])
index = ['a', 'b', 'd', 'c', 'e', 'f', 'g', 'g', 'h', 'i']
df = pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]},
                  index=index)
print(df)

print(df.loc['g'])
print(df.loc['c':])
print(df.loc[['g', 'i'], ['a', 'c']])
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
print(df[df['a'] >= 3])
print(df[df['a'] >= 3][['a', 'c']])
print(df[(df['a'] >= 3) & (df['b'] < 16)])
a = (df['a'] >= 3) & (df['b'] < 16)

print(df[(df['a'] >= 3) & ((df['b'] < 16) | (df['c'] == 30))])


df.sort_index(ascending=False, inplace=True)
print(df)

print(df.reset_index())
print(df.reset_index(drop=True))
df.reset_index(drop=True, inplace=True)
print(df)
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

df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})

df.dropna(inplace=True)
df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})
print(df.dropna(axis=1))
df.dropna(axis=1, inplace=True)
df = pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})

df.fillna(0, inplace=True)

df = pd.DataFrame({'a': [1, 1, 3, 4, np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 7, 6, 4]})
print(df.fillna(method='bfill'))
print(df.fillna(method='ffill'))
# limit 설정
print(df.fillna(method='ffill', limit=1))
df = pd.DataFrame({'a': [1, 1, 3, 4, np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 7, 6, 4]})

df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

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
df.astype({'판매량': 'int'})
print(df.dtypes)
df = df.astype({'판매량': 'int'})
print(df.dtypes)
df['판매량 보정'] = df['판매량'] + 1
df['방문자수'] = pd.to_numeric(df['방문자수'], errors='coerce')
print(df.dtypes)
print(df))
df.fillna(0, inplace=True)
df = df.astype({'방문자수': 'int'})
print(df)
print(df.dtypes)
df['판매일'] = pd.to_datetime(df['판매일'], format='%m/%d/%y')
print(df)
print(df.dtypes)

df = pd.DataFrame({
    'a': [13, 1, 3, 4],
    'b': [2, 4, 2, 3],
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
