# CH03_10. 레코드, 칼럼 추가 삭제

import pandas as pd
df=pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, 2, 3, 4], 'c': [3, 4, 7, 6, 4]})
df

#칼럼 추가
#문제: 1, 3, 6, 4, 8로 이루어진 d 칼럼을 추가하시오.
df['d']=[1, 3, 6, 4, 8]
df

#문제: 1로 이루어진 e 칼럼을 추가하시오.
df['e']=[1, 1, 1, 1, 1]
df['e']=1 #리스트로 넣으면 안됨. 모든 값이 1
df.dtypes

#문제: a+b-c의 결과로 이루어진 f 칼럼을 추가하시오.
df['f']=df['a']+df['b']-df['c']

#칼럼 삭제
#문제: 칼럼 d, e, f를 삭제하시오.
df.drop(['d', 'e', 'f'], axis=1) #'axis=1'을 추가해줘야 칼럼이 삭제. '=1'이 수식으로서는 무의미한듯
df.drop(['d', 'e', 'f'], inplace=True, axis=1) 
df

#레코드 추가
#문제: a에는 6, b에는 7, c에는 8을 추가하시오.
df.append({'a':6, 'b':7, 'c':8}, ignore_index=True)
df.append({'a':6, 'b':7, 'c':8}, ignore_index=True, inplace=True) #append는 inplace 안됨
df=df.append({'a':6, 'b':7, 'c':8}, ignore_index=True)
df

#문제: a에는7, b에는 8, c에는 9를 추가하시오.
df.loc[6]=[7, 8, 9] #loc, iloc 다음에는 위치값
df

#레코드 삭제
#문제: 첫 번째 레코드를 삭제하시오.
df
df.drop(0)

#문제: 첫 번째, 두 번째 레코드를 삭제하시오.
df=df.drop([0, 1])
df

#문제: 첫 번째에서 네 번째 레코드를 삭제하시오.
df=pd.DataFrame({'a': [1, 1, 3, 4, 5], 'b': [2, 3, 2, 3, 4], 'c': [3, 4, 7, 6, 4]})
df.drop([i for i in range(4)]) #0~3까지는 삭제. list comprehension
df.drop(df.index[:4]) #indexing

#문제: a가 4 미만인 레코드들을 삭제하시오.
df
df[df['a']<4].index #index = 로우명, 칼럼 a에서 4보다 작은 index 출력
df.drop(df[df['a']<4].index) #출력된 index '0', '1', '2' drop

#문제: a가 3 미만이고, c가 4인 레코드들을 삭제하시오.
df
df[(df['a']<3) & (df['c']==4)]
df[(df['a']<3) & (df['c']==4)].index
df.drop(df[(df['a']<3) & (df['c']==4)].index)