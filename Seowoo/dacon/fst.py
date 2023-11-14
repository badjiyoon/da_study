import pandas as pd
import numpy as np
import statsmodels.api as sm

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 가중치 초기화
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

        # 편향 초기화
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            # Forward Pass
            hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            predicted_output = self.sigmoid(output_layer_input)

            # Calculate error
            output_error = targets - predicted_output

            # Backward Pass
            output_delta = output_error * self.sigmoid_derivative(predicted_output)
            hidden_error = output_delta.dot(self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += hidden_layer_output.T.dot(output_delta)
            self.weights_input_hidden += inputs.T.dot(hidden_delta)
            self.bias_output += np.sum(output_delta, axis=0, keepdims=True)
            self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True)

    def predict(self, inputs):
        # Forward Pass
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        predicted_output = self.sigmoid(output_layer_input)

        return predicted_output

# Excel 파일 경로
excel_file_path = '../../data/train_test.xlsx'

df = pd.read_excel(excel_file_path, engine='openpyxl')

data = pd.DataFrame(df)

print(data.columns)
# 특정 코드 목록

tgList = df.loc[df['ID'].str.startswith('TG')]   # TG : 감귤
bcList = df.loc[df['ID'].str.startswith('BC')]   # BC : 브로콜리
rdLit = df.loc[df['ID'].str.startswith('RD')]   # RD : 무
crList = df.loc[df['ID'].str.startswith('CR')]   # CR : 당근
cbList = df.loc[df['ID'].str.startswith('CB')]   # CB : 양배추


tgList = pd.get_dummies(tgList, columns=['corporation', 'location'])
# ID제거
tgList = tgList.drop(['timestamp', 'ID', 'item'], axis=1)

for column_name in tgList.columns:
    if tgList[column_name].dtype == bool:
        # bool인 경우 int로 변환
        tgList[column_name] = tgList[column_name].astype(int)

tgList_input = tgList[tgList.columns.difference(['price(??kg)'])]
tgList_target = tgList[['price(??kg)']]
print(tgList_input.shape[1])
print(tgList_target)
# 신경망 생성
input_size = tgList_input.shape[1]
hidden_size = 4
output_size = tgList_target.shape[1]
epochs = 10

model = NeuralNetwork(input_size, hidden_size, output_size)

# 학습
model.train(tgList_input, tgList_target, epochs)

# 예측
predicted_output = model.predict(tgList_input)
print("Predicted Output:")
print(predicted_output)



#
# new_order = [col for col in tgList.columns if col != 'price(??kg)'] + ['price(??kg)']
# tgList = tgList[new_order]
# print(tgList)
#
# forecast_steps = 10
# forecast_tgList = pd.DataFrame({'timestamp': pd.date_range(start=tgList['timestamp'].iloc[-1], periods=forecast_steps + 1, freq='D')[1:]})
#
#
# order = (1, 1, 1)  # SARIMA(p, d, q, s): Seasonal order (p, d, q) 및 s는 계절성 주기
# # 모든 컬럼에 대해 SARIMAX 모델 훈련 및 예측
#
# model = sm.tsa.SARIMAX(tgList['price(??kg)'], order=order, seasonal_order=(0, 0, 0, 0))
# results = model.fit()
#
# # 예측 결과를 데이터프레임에 추가
# forecast_values = results.get_forecast(steps=forecast_steps)
# forecast_tgList[f'forecast_price(??kg)'] = forecast_values.predicted_mean.values
#
# # 결과 출력
# df = pd.concat([df, forecast_tgList.iloc[:, 1:]], axis=1)
# print(df)
