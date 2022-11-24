# CH04_08. lineplot 을 이용한 선도표 그리기
#lineplot: scatterplot처럼 된 점들을 이은 그래프

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})
import pandas as pd

df = sns.load_dataset('flights')
df.head()
df.isnull().sum()

#선도표 그리기
sns.lineplot(data = df, x = 'year', y = 'passengers')
plt.show()

#xtick 에 전체 년도 표시
sns.lineplot(data = df, x = 'year', y = 'passengers')
plt.xticks(df['year']); #x축의 year로 생략없이
#실즈?가 데이터프레임 형태? 무슨 말이지?
plt.show()

#xtick 기울기 추가
sns.lineplot(data = df, x = 'year', y = 'passengers')
plt.xticks(df['year'], rotation = 45); #x축값들이 45도 기울여져서 나타남
plt.show()

#월별로 그리기
sns.lineplot(data = df, x = 'year', y = 'passengers', hue = 'month')
#범례는 legend인줄 알았는데 hue옵션 넣으니 나옴
plt.show()

#색상 변경
sns.lineplot(data = df, x = 'year', y = 'passengers', hue = 'month', palette = 'Set2')
#set2라는 종류의 색상 팔레트
plt.show()

#선 표현 방법 변경
sns.lineplot(data = df, x = 'year', y = 'passengers', hue = 'month', palette = 'Set2', style = 'month')
#선 종류에 따라서 month를 다르게 표현
plt.show()