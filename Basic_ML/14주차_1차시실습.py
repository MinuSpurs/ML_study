#%%

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 
# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pyplot as plt
# sklearn으로부터 데이터를 불러온다.
import sklearn
from sklearn.datasets import load_boston

boston = load_boston()
x_data, y_data = boston.data, boston.target
# data scaling
x_data = sklearn.preprocessing.scale(x_data, axis=0)
x_data.shape
y_data.shape

#%%

# check function
tau = 0.8
t = np.linspace(-3, 3, num=100, endpoint=True)
plt.plot(t, np.max((tau*t, (tau-1)*t), axis=0), '-', color='blue', linewidth=4)
plt.axvline(x=0, linestyle='--', c='black', linewidth=3)

#%%

# 분위수 값 설정
tau = 0.2
input_layer = layers.Input((x_data.shape[1]))
dense_layer1 = layers.Dense(5, activation='sigmoid')
x = dense_layer1(input_layer)
dense_layer2 = layers.Dense(1, activation='linear')
x = dense_layer2(x)

model = K.models.Model(input_layer, x)
model.summary()

#%%

optimizer = K.optimizers.SGD(lr=0.5)

#%%

# 사용자 지정 손실함수
def quantile_loss(q, y, y_pred):
    r = y - y_pred
    l = tf.reduce_mean(tf.maximum(tau*r, (tau-1)*r), axis=0, keepdims=False)
    return l

#%%

epochs = 500
batch_size = 128
for epoch in range(epochs):
    idx = np.random.randint(0, len(y_data), batch_size)
    x_batch = tf.cast(x_data[idx, :], tf.float32)
    y_batch = tf.cast(y_data[idx][:, np.newaxis], tf.float32)
    
    with tf.GradientTape() as tape:
        y_hat = model(x_batch)
        loss = quantile_loss(tau, y_batch, y_hat)
    
    grad = tape.gradient(loss, model.weights)
    optimizer.apply_gradients(zip(grad, model.weights))
    
    if epoch % 10 == 0:
        print('({} epoch: loss: {:.6})'.format(epoch, loss.numpy()[0]))
        
#%%

# prediction
room = np.linspace(np.min(x_data[:, 5]), np.max(x_data[:, 5]), 100)
new_data = np.zeros((100, x_data.shape[1]))
new_data[:, 5] = room
# 신경망 모형 예측
nn_pred = model(tf.cast(new_data, tf.float32))

# 선형회귀모형을 사용할 수 있는 모듈을 불러온다.
import statsmodels.api as sm
# 단순 선형회귀모형 적합과 예측
simple_reg = sm.OLS(y_data, sm.add_constant(x_data)).fit()
simple_reg_pred = simple_reg.predict(sm.add_constant(new_data))

# 시각화
plt.figure(figsize=(9, 6))
plt.plot(room, np.squeeze(nn_pred.numpy()), '-', color='red', linewidth=4)
plt.plot(room, simple_reg_pred, '--', color='blue', linewidth=4)

#%%
# K.backend.clear_session()
#%%

# prediction[:5]
# sort_idx = np.argsort(np.squeeze(prediction.numpy()))
# plt.figure(figsize=(9, 6))
# plt.plot(y_data[sort_idx], 'o', color='blue', markersize=5)
# plt.plot(np.squeeze(prediction.numpy())[sort_idx], '-', color='red', linewidth=4)

#%%
