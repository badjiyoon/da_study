# CH04_11. boxplot 을 이용한 상자 수염 그림 그리기

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})
import pandas as pd

df = sns.load_dataset('iris')

#boxplot 그리기: 가장 많은 25% 구간만 가운데 박스로 나타냄
sns.boxplot(data = df)
plt.show()

#세로방향으로 그림 크기 키우기
plt.subplots(figsize = (7, 8))
sns.boxplot(data = df)
plt.show()

#가로 방향으로 그리기
sns.boxplot(data = df, orient = 'h') #orient: 방향
plt.show()

#swarmplot 을 이용한 raw data 확인 (swarm: 떼, 무리)
plt.subplots(figsize = (10, 10))
sns.swarmplot(data = df)
plt.show()

#겹쳐 그리기
plt.subplots(figsize = (10, 10))
sns.boxplot(data = df)
sns.swarmplot(data = df)
plt.show()

#swarmplot 색상 조정
plt.subplots(figsize = (10, 10))
sns.boxplot(data = df)
sns.swarmplot(data = df, color = 'black') #swamplot에만 검정색 옵션
plt.show()

#violinplot
sns.violinplot(data = df) #바이올린같이 생김
plt.show()

#stripplot
sns.stripplot(data = df) #그냥 점 찍힌 느낌
plt.show()