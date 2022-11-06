# CH04_09. heatmap 을 이용한 히트맵 그리기

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})
import pandas as pd

df = sns.load_dataset('flights')
df.head()
df.describe()
df.isnull().sum()

#문제 : 각 년도별 월별 승객수를 히트맵으로 나타내시오.

#pivot을 사용한 데이터 형태 변경
pivot_data = df.pivot("month", "year", "passengers")
pivot_data
sns.heatmap(pivot_data)
plt.show()

#구분선 추가
sns.heatmap(pivot_data, linewidths=.5)
plt.show()

#cmap 을 통한 colorbar 색상 변경
sns.heatmap(pivot_data, cmap="Blues") #cmap: chart 스타일? 색상 변경
plt.show()

#수치 입력
sns.heatmap(pivot_data, cmap="Blues", annot = True) #annot=annotation 노테이션 달아주기
plt.show()

#정수 형태로 출력
sns.heatmap(pivot_data, cmap="Blues", annot = True, fmt = 'd') #fmt=format, d=정수
plt.show()

#colorbar 가로로 놓기
fig, (ax, cbar_ax) = plt.subplots(2)
plt.show()

ax = sns.heatmap(pivot_data, ax=ax,
                  cbar_ax=cbar_ax, 
                  cbar_kws={"orientation": "horizontal"}) #orientation: 방향, horizontal: 수평
#에러남..

#orientation : vertical
ax = sns.heatmap(pivot_data, ax=ax,
                  cbar_ax=cbar_ax,
                  cbar_kws={"orientation": "vertical"})

#colorbar 와 히트맵 같의 비율 조정
grid_kws = {"height_ratios": (.85, .1), "hspace": 0.4}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax = sns.heatmap(pivot_data, ax=ax,
                 cbar_ax=cbar_ax,
                 cbar_kws={"orientation": "horizontal"})

#최종
grid_kws = {"height_ratios": (.85, .1), "hspace": 0.4}

f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
ax = sns.heatmap(pivot_data, ax=ax,
                 cbar_ax=cbar_ax,
                 cbar_kws={"orientation": "horizontal"},
                 cmap="Blues", 
                 annot = True, fmt = 'd')