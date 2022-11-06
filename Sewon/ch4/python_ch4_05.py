# CH04_05. barplot을 이용한 막대 그래프 그리기

import matplotlib.pyplot as plt 
import matplotlib as mpl
plt.rcParams['font.family']=['NanumGothic','sans-serif'] #sans-serif가 뭐지?
plt.rcParams['axes.unicode_minus']=False
import seaborn as sns
import pandas as pd

#문제 : 다음은 2021년 영화별 관람자 수다. 이를 막대그래프로 표현하시오.
movie_title = ['크루엘라', '극장판 귀멸의 칼날: 무한열차편', '학교 가는 길']
audience = [664308, 2099131, 20067]

data1 = {'영화제목' : movie_title, '누적관객' : audience}
df = pd.DataFrame(data1)
df

#barplot: 막대그래프
sns.barplot(data = df, x = '영화제목', y = '누적관객') #'data='부분은 변수명 아니고 옵션
plt.show()

#차트 크기 변경
sns.barplot(data = df, x = '영화제목', y = '누적관객') 
sns.set(rc={'figure.figsize':(10, 5)}) #rc가 뭘까
plt.show()

#누적관객수 별로 그리기
sns.barplot(data = df, x = '영화제목', y = '누적관객',
             order = df.sort_values('누적관객').영화제목)
plt.show() #1e6 = 0.25*10^6

#ascending 을 이용한 내림차순 정렬
sns.barplot(data = df, x = '영화제목', y = '누적관객',
             order = df.sort_values('누적관객', ascending = False).영화제목)
#ascending=True: 오름차순, ascending=False: 내림차순
plt.show()

#관객 수 포맷 변환
chart = sns.barplot(data = df, x = '영화제목', y = '누적관객',
             order = df.sort_values('누적관객', ascending = False).영화제목)
ylabels = ['{:,.0f}'.format(i) + '만 명' for i in chart.get_yticks() / 10000] #다시 알아볼 것
chart.set_yticklabels(ylabels)
plt.show()

'''
<stdin>:1: UserWarning: FixedFormatter should only be used together with FixedLocator
<stdin>:1: 사용자 경고: FixedFormatter는 FixedLocator와 함께 사용해야 합니다.
보기 안좋다는 뜻
'''

#경고문 제외 (에러가 불편할 경우 작성)
import matplotlib.ticker as mticker

chart = sns.barplot(data = df, x = '영화제목', y = '누적관객',
             order = df.sort_values('누적관객', ascending = False).영화제목)

ticks_labels = chart.get_yticks().tolist()
chart.yaxis.set_major_locator(mticker.FixedLocator(ticks_labels))
chart.set_yticklabels(['{:,.0f}'.format(i / 10000) + '만 명' for i in ticks_labels])
plt.show()

#가로로 그리기
chart = sns.barplot(data = df, x = '누적관객', y = '영화제목',
             order = df.sort_values('누적관객', ascending = False).영화제목)

xlabels = ['{:,.0f}'.format(i) + '만 명' for i in chart.get_xticks() / 10000]
chart.set_xticklabels(xlabels) #y라벨즈 대신 x라벨즈로 바꾸기만 하면 barplot에서 알아서 가로로 바꿔줌
plt.show()

#차트 색상 변경
chart = sns.barplot(data = df, x = '누적관객', y = '영화제목',
             order = df.sort_values('누적관객', ascending = False).영화제목,
             color = 'blue') #color 옵션만 추가

xlabels = ['{:,.0f}'.format(i) + '만 명' for i in chart.get_xticks() / 10000]
chart.set_xticklabels(xlabels)
plt.show()

#제목 추가, 폰트 사이즈 변경
chart = sns.barplot(data = df, x = '누적관객', y = '영화제목',
             order = df.sort_values('누적관객', ascending = False).영화제목,
             color = 'pink')

xlabels = ['{:,.0f}'.format(i) + '만 명' for i in chart.get_xticks() / 10000]
chart.set_xticklabels(xlabels)

plt.xlabel('누적관객', fontsize = 15)
plt.ylabel('영화제목', fontsize = 15)
plt.title('영화 별 누적관객수', fontsize = 20)
plt.show()