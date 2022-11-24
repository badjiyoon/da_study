import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load_dataset
df = sns.load_dataset('iris')
print(df.head())
# 산점도 그래프 그리기
# plt.scatter(df['petal_length'], df['petal_width'])
# 시본을 사용한 산점도 차트
# sns.scatterplot(data=df, x='petal_length', y='petal_width')
# matplotlib 호환
# sns.scatterplot(data=df, x='petal_length', y='petal_width')
# plt.title('iris')

plt.show()
