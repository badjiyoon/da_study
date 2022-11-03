# CH03_01. 데이터 프레임 생성

import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
type(df)
df #데이터 프레임 출력

#Dict를 통한 데이터 프레임 생성 -열 순서로 작성
dummy = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
df2 = pd.DataFrame(dummy)
df2

#List를 이용한 데이터 프레임 생성 -행 순서로 작성
a= [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
df3 = pd.DataFrame(a)
df3

df3.columns=['a', 'b', 'c'] #열 이름 정해줌
df3

df3.rows=['e', 'f', 'g'] #행은 없네
df3

#문제: 아래 테이블과 같은 데이터 프레임을 만드시오.
a={'company': ['abc', '회사', '123'], '직원수': [400, 10, 6]}
df4 = pd.DataFrame(a)
df4

#문제: 아래 테이블과 같은 데이터 프레임을 만드시오.
# a={'company': ['abc', '회사', '123'], '직원수': [400, 10, 6], '위치': ['Seoul', NaN, 'Busan']}
# a={'company': ['abc', '회사', '123'], '직원수': [400, 10, 6], '위치': ['Seoul', , 'Busan']}

'''
NaN은 결측값인데, 그냥 넣으면 변수로 취급해서 안됨
공란/띄어쓰기로 넣어도 안됨
numpy 라이브러리를 import 해서 넣을 수 있음
'''

import numpy as np
a={'company': ['abc', '회사', '123'], '직원수': [400, 10, 6], '위치': ['Seoul', np.NaN, 'Busan']}
df5 = pd.DataFrame(a)
df5 #표가 너무 엉망으로 생김..

