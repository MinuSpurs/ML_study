#%%

'''회귀모형1'''

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 
# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pylab as plt 

np.random.seed(1)
n = 100
p = 10
x = np.random.normal(size=(n, p))
y = np.sin(x[:, [4]])*10 + np.random.normal(size=(n, 1))

fig, ax = plt.subplots(1,1, figsize=(5,5))
plt.plot(x[:, [4]], y, 'o', color='blue')

#%%

input_layer = layers.Input((p))
dense_layer1 = layers.Dense(units=2, activation='sigmoid')
h = dense_layer1(input_layer)
dense_layer2 = layers.Dense(units=1, activation='linear')
h = dense_layer2(h)

model = K.models.Model(input_layer, h)
model.summary()

#%%

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['mse'])

#%%

model.fit(x, y,
          epochs=200,
          validation_split=0.2)
yhat = model.predict(x)
plt.plot(y, yhat, 'o', color='blue')

#%%

dense_layer1.weights
dense_layer1.weights[0][0, 0]
dense_layer1.weights[0][1, 0]

#%%

'''회귀모형2'''

np.random.seed(1)
n = 10000
p = 10
x = np.random.normal(size=(n, p))
y1 = np.sin(x[:, [4]])*10 + np.random.normal(size=(n, 1))
y2 = np.cos(x[:, [3]])*15 + np.random.normal(size=(n, 1))
y3 = np.cos(x[:, [0]])*13 + np.random.normal(size=(n, 1))
y = np.hstack((y1, y2, y3))

#%%

input_layer = layers.Input((p))
dense_layer1 = layers.Dense(units=10, activation='sigmoid')
h = dense_layer1(input_layer)
dense_layer2 = layers.Dense(units=3, activation='linear')
h = dense_layer2(h)

model = K.models.Model(input_layer, h)
model.summary()

#%%

model.compile(optimizer='sgd',
              loss='mse',
              metrics=['mse'])

#%%

model.fit(x, y,
          epochs=100,
          validation_split=0.2)

#%%

yhat = model.predict(x)
for i in range(y.shape[1]):
    plt.plot(y[:, i], yhat[:, i], 'o', color='blue')
    plt.show()

#%%

dense_layer1.weights[0][:5]
dense_layer1.weights[1]
dense_layer2.weights[0]
dense_layer2.weights[1]

#%%

'''분류모형'''

from sklearn.datasets import load_wine
data = load_wine()
x = data.get('data')
y = data.get('target')
y = K.utils.to_categorical(y, num_classes=3, dtype='float32')

#%%

input_layer = layers.Input((x.shape[1]))
dense_layer1 = layers.Dense(units=10, activation='relu')
h = dense_layer1(input_layer)
dense_layer2 = layers.Dense(units=3, activation='softmax')
h = dense_layer2(h)

model = K.models.Model(input_layer, h)
model.summary()

#%%

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%

model.fit(x, y,
          epochs=50,
          validation_split=0.2)

#%%

yhat = model.predict(x)
yhat[:5] 
model.evaluate(x, y, verbose=0)

#%%

import sklearn
x = sklearn.preprocessing.scale(x, axis=0)

input_layer = layers.Input((x.shape[1]))
dense_layer1 = layers.Dense(units=10, activation='relu')
h = dense_layer1(input_layer)
dense_layer2 = layers.Dense(units=3, activation='softmax')
h = dense_layer2(h)
model = K.models.Model(input_layer, h)
model.summary()
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x, y,
          epochs=50,
          validation_split=0.2)

#%%

yhat = model.predict(x)
yhat[:5] 
model.evaluate(x, y, verbose=0)

#%%
# K.backend.clear_session()
#%%
