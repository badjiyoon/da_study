#CH05_07. 딥러닝 모델 구성 및 결과 검증

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

x_train[0]
x_train[0].shape #x_train의 형태 출력

#정규화
x_train, x_test = x_train / 255.0, x_test / 255.0 
'''
이미지 데이터이므로 0~255 사이의 숫자로 구성
숫자가 너무 크면 과적합 우려
255로 나누어 최대값이 1이 되도록 정규화
'''

#그림 그리기
plt.imshow(x_test[0])
plt.show()
y_test[0]

#모델 작성
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #28*28인 데이터가 평평하게 들어감(?)
  tf.keras.layers.Dense(128, activation='relu'), #히든레이어 128개, 활성화함수: relu
  tf.keras.layers.Dropout(0.2), #20% 정도는 생략, 불충분한 데이터가 들어와도 감안하도록 함
  tf.keras.layers.Dense(10, activation='softmax') #출력은 10가지로 분류, 활성화함수: softmax
])

model.compile(optimizer='adam', #optimizer: 학습의 효율을 높여줄 수 있도록 도와주는 함수, 최적화
              loss='sparse_categorical_crossentropy', #loss=실제값-예측값, 어떻게 정의할지는 모델에 따라 다름
              metrics=['accuracy']) #metrics: 우리의 목표(?)

#모델 시각화
model.summary()

#모델 학습 및 평가
model.fit(x_train, y_train, epochs = 5) #모델 학습
model.evaluate(x_test,  y_test, verbose = 2) #모델 평가
'''
optimizer: 학습의 효율을 높여줄 수 있도록 도와주는 함수, 최적화
loss=실제값-예측값, 어떻게 정의할지는 모델에 따라 다름
epoch: 몇 번이나 반복하여 학습할지
'''