# CH04_12. displot / histplot 을 이용한 히스토그램 그리기

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15, 5)})
import pandas as pd

df = sns.load_dataset('iris')

#distplot: 곧 사라질 예정
sns.distplot(df['sepal_length'])
plt.show()

#구간 개수 설정
sns.distplot(df['sepal_length'], bins = 15) #세로 막대가 15개
plt.show()

#histplot: 그냥 막대그래프
sns.histplot(df['sepal_length'])
plt.show()

#구간 개수 설정
sns.histplot(df['sepal_length'], bins = 20)
plt.show()

#kde 추가 
'''
KDE: 커널 밀도 함수
Kernel Density Estimator
'''
sns.histplot(df['sepal_length'], bins = 20, kde = True)
plt.show()

#count -> density
sns.histplot(df['sepal_length'], bins = 20, kde = True, stat = 'density')
#stat: y축 이름을 density
plt.show()

sns.histplot(df['sepal_length'], bins = 20, kde = True, stat = 'density', hue = 'species')
plt.show()

'''
hue: species 할 수 없는 이유
hue는 분리하는 건데, 현재 데이터를 sepal length에 대해서만 가져왔기 때문에 분리가 안됨
데이터를 전체 가져와서 그 중에 x축과 y축을 분리해야 함
'''

sns.histplot(data = df, x = 'sepal_length', bins = 20, kde = True, stat = 'density', hue = 'species')
plt.show()

#x 축, y 축 변경 (x, y 교체)
sns.histplot(data = df, y = 'sepal_length', bins = 20, kde = True, stat = 'density', hue = 'species')
plt.show()

#차원 추가
sns.histplot(data = df, y = 'sepal_length', x = 'sepal_width')
plt.show()

#칼라바 추가
sns.histplot(data = df, y = 'sepal_length', x = 'sepal_width', cbar = True)
#cbar: 오른쪽에 칼라바 추가됨
plt.show()

#displot
sns.displot(df['sepal_length'])
plt.show()

sns.displot(data = df, x = 'sepal_length')
plt.show()

#그림 크기 변경
sns.displot(data = df, x = 'sepal_length', height = 5, aspect = 3)
plt.show()

'''
height: 높이
aspect: 막대 한 개의 가로 길이
width: 막대그래프 가로 길이
height * aspect = width
'''

#histgram 과 한 옵션 사용(1)
a = sns.displot(data = df, y = 'sepal_length', bins = 20, kde = True, stat = 'density', hue = 'species')
plt.show()

#histgram 과 한 옵션 사용(2)
sns.displot(data = df, y = 'sepal_length', x = 'sepal_width', cbar = True)
plt.show()

#jointplot
sns.jointplot(data = df,  x = 'sepal_width', y = 'sepal_length')
plt.show()

#histplot 의 형식으로 변경
sns.jointplot(data = df,  x = 'sepal_width', y = 'sepal_length', kind = 'hist', cbar = True)
plt.show()

sns.jointplot(data = df,  x = 'sepal_width', y = 'sepal_length', kind = 'hist', cbar = True, height = 10)
plt.show()