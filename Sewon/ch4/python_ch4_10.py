# CH04_10. pie 를 이용한 원형차트 그리기

import matplotlib.pyplot as plt
plt.rcParams['font.family']=['NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

movie_title = ['크루엘라', '극장판 귀멸의 칼날: 무한열차편', '학교 가는 길']
audience = [664308, 2099131, 20067]

data = {'영화제목' : movie_title, '누적관객' : audience}
df = pd.DataFrame(data)

#pie chart 생성
plt.pie(df['누적관객'])
plt.show()

#라벨 추가
plt.pie(df['누적관객'], labels = df['영화제목']); 
plt.show()

#값 추가
plt.pie(df['누적관객'], labels = df['영화제목'], autopct = '%0.2f%%'); 
#autopct: 자동으로 퍼센트로 계산해줌 (%0.2f%%: 소수 둘째 자리까지 나타내기)
plt.show()

#colors 를 이용한 색상 변경하기
colors_list = ['#F08080', '#DCDCDC', '#FFF8DC'] #colors_list = ['red', 'green', 'blue']

plt.pie(df['누적관객'], labels = df['영화제목'], autopct = '%0.2f%%', colors = colors_list);
plt.show()

#explode 를 이용한 중심으로부터 그림 떼어내기
explode_list = [0, 0.1, 0] #두 번째 데이터만 떼어내서 강조

plt.pie(df['누적관객'], labels = df['영화제목'], autopct = '%0.2f%%', explode = explode_list);
plt.show()