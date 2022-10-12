# CH02_14. 외부 라이브러리

#라이브러리 다운로드
'''
명령 프롬프트에서?
!pip install pandas
'''

#라이브러리 불러오기
import pandas
pandas.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

#약어를 사용하여 라이브러리 불러오기
import pandas as pd
pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

#모듈 지우기
del pd
pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
