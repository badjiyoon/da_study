import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

sns.set(rc={'figure.figsize': (10, 5)})

df = sns.load_dataset('iris')
print(df.head())

# sns.boxplot(data=df)

# 세로방향으로 그림 크기 키우기
# plt.subplots(figsize=(7, 8))
# sns.boxplot(data=df)
# 가로뱡향으로 그리기
# sns.boxplot(data=df, orient='h')
# swarmplot을 이용한 ROW DATA확인
# plt.subplots(figsize=(10, 10))
# sns.swarmplot(data=df)
# 겹쳐 그리기
# plt.subplots(figsize=(10, 10))
# sns.boxplot(data=df)
# sns.swarmplot(data=df)
# sns.swarmplot(data=df, color='black')
# sns.violinplot(data=df)
sns.stripplot(data=df)
plt.show()
