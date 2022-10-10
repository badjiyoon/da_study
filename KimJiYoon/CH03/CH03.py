import pandas as pd
import numpy as np

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


