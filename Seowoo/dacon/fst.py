import pandas as pd

# Excel 파일 경로
excel_file_path = '../../data/train_test.xlsx'

df = pd.read_excel(excel_file_path, engine='openpyxl')

data = pd.DataFrame(df)

print(data.columns)
# 특정 코드 목록
target_code = 'TG'

# 각 코드에 대한 행 추출하여 리스트 생성
filtered_rows = []

filtered_rows = df.loc[df['ID'].str.startswith(target_code)]

print(filtered_rows)
