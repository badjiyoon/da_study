import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))
print(x_train)
print(x_train[0].shape)
# 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0
# 그림 그리기
# plt.imshow(x_test[0])
print(y_test[0])
# plt.show()
# 모델 작성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(129, activation='relu'),
    tf.keras.layers.Dropout(0, 2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.complie(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)
