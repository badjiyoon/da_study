import pandas as pd

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

df3.columns = ['a', 'b', 'c']
print(df3)

