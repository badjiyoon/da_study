# CH03_12. 데이터프레임 결합_상하결합

import pandas as pd

df1=pd.DataFrame({'A': [1, 2, 3], 'B': [11, 12, 13], 'C': [21, 22, 23]})
df2=pd.DataFrame({'A': [4, 5, 6], 'B': [14, 15, 16], 'C': [24, 25, 26]})

pd.concat([df1, df2]) #concatenate: 연관시키다
pd.concat([df2, df1])

#index 초기화를 위해서는 ignore_index=True
pd.concat([df1, df2], ignore_index=True) 

#필드의 순서가 섞였을 때 결합 결과 확인
df1=pd.DataFrame({'A': [1, 2, 3], 'B': [11, 12, 13], 'C': [21, 22, 23]})
df2=pd.DataFrame({'B': [14, 15, 16], 'A': [4, 5, 6], 'C': [24, 25, 26]})
df1
df2
pd.concat([df1, df2]) #칼럼의 순서가 아닌 칼럼명으로 구분

#서로 다른 필드로 구성되어 있는 데이터프레임의 결합
df1=pd.DataFrame({'A': [1, 2, 3], 'B': [11, 12, 13], 'C': [21, 22, 23], 'D': [31, 32, 33]})
df2=pd.DataFrame({'A': [4, 5, 6], 'B': [14, 15, 16], 'C': [24, 25, 26], 'E': [41, 42, 43]})
pd.concat([df1, df2]) #아무 옵션도 없으면 outer join
pd.concat([df1, df2], join='outer') #outer join=합집합
pd.concat([df1, df2], join='inner') #inner join=교집합