import pandas as pd

# df = pd.read_csv('../../comFiles/과일가게.csv')
# print(type(df))
# print(df.head())
# print(df.tail())

# 첫 번쨰 열을 인덱스로열로 삼고 싶을 경우 (옵션처리)
# df = pd.read_csv('../../comFiles/과일가게.csv', index_col=0)
# print(df.head())

# 구분자가, 가 아닌 다른 기호인 경우
# df = pd.read_csv('../../comFiles/read_sep.txt', index_col=0, sep='|')
# print(df)

# header 옵션
# df = pd.read_csv('../../comFiles/read_multi_header.csv', header=1)
# print(df)
# print(df.columns)

# 데이터를 읽으면서 컬러명을 추가하고 싶을떄
# df = pd.read_csv('../../comFiles/make_column_name.csv', index_col=0, names=['품목', '크기', '금액', '수수료'])
# print(df)

df = pd.read_csv('../../comFiles/과일가게.csv', usecols=['품목', '크기'])
# print(df.head())

# 파일저장
df.to_csv('../../comFiles/make_csv.csv')
