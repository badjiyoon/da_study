# CH03_02. 칼럼명 추출 변경

import pandas as pd

df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df

#칼럼명 얻기 -인덱스
df.columns[1]

#칼럼명인 a, b, c를 d, e, f로 바꾸어라.
df.columns = ['d', 'e', 'f']
df

#칼럼명인 d, e, f 중 d를 '디'로, f를 '에프'로 바꾸어라.
df.columns = ['디', 'e', '에프']
df

#rename을 통한 칼럼명 변경
df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df.columns = ['d', 'e', 'f']

df.rename(columns={'d':'디', 'f':'에프'}) #보여질 때는 이름이 바뀌었지만, 실제 데이터명이 바뀌진 않음
df.rename(columns={'d':'디', 'f':'에프'}, inplace=True) #inplace=True로 되어 있어야 저장됨
df


