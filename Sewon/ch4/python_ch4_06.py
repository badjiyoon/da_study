# CH04_06. countplot을 이용한 막대 그래프 그리기

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})
import pandas as pd

#titanic 데이터 셋의 성별 인원수를 시각화 하시오.
df = sns.load_dataset('titanic')
df.head() #간략하게 데이터셋 확인
df.describe() #수치 데이터의 기술 통제? 

df.isnull().sum() #결측치 처리
df.groupby(by = 'sex')['sex'].count() #성별로 그룹핑해서 카운트

#countplot 을 사용한 해결
sns.countplot(data = df, x = 'sex')
plt.show()

#가로로 그리기
sns.countplot(data = df, y = 'sex')
plt.show()

#문제 : titanic 데이터 셋의 성별 인원수를 객실 등급별로 시각화 하시오.
df.head()
df[['sex', 'class']]
df.groupby(by = ['sex', 'class'])['sex'].count()
'''
sex     class
female  First      94
        Second     76
        Third     144
male    First     122
        Second    108
        Third     347
Name: sex, dtype: int64
'''

sns.countplot(data = df, x = 'sex', hue = 'class') #hue: 카테고리별로 분류
plt.show()

#palette 를 사용한 색상 조정 (옵션 찾아서 활용)
sns.countplot(data = df, x = 'sex', hue = 'class', palette = 'flare')
plt.show()