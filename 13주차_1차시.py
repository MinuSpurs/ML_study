#%%

'''introduction'''

# mnist dataset을 제공하는 tensorflow 모듈을 불러온다.
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 
# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pylab as plt 

# mnist dataset을 읽어온다.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = K.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = K.utils.to_categorical(y_test, num_classes=10, dtype='float32')

#%%

# grayscale의 colormap을 이용해 sample을 그린다.
plt.imshow(x_train[0], cmap='gray') 

#%%

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((len(x_train), 784))
x_train 
x_test = x_test.reshape((len(x_test), 784))
x_test

#%%

input_layer = layers.Input((784))
dense_layer = layers.Dense(20, activation='relu')
x = dense_layer(input_layer)
softmax_layer = layers.Dense(10, activation='softmax')
x = softmax_layer(x)

model = K.models.Model(input_layer, x)
model.summary()

#%%

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=128,
          epochs=30,
          validation_split=0.2)

#%%

model.evaluate(x_test, y_test, verbose=False)
pred_y = model.predict(x_test)
idx = np.where(np.equal(np.argmax(pred_y, axis=1), np.argmax(y_test, axis=1)))[0]
plt.imshow(x_test[idx[0]].reshape(28, 28), cmap='gray') 
pred_y[idx[0]]
y_test[idx[0]]
np.argmax(pred_y[idx[0]])
np.argmax(y_test[idx[0]])

#%%

idx = np.where(np.logical_not(np.equal(np.argmax(pred_y, axis=1), np.argmax(y_test, axis=1))))[0]
plt.imshow(x_test[idx[0]].reshape(28, 28), cmap='gray') 
pred_y[idx[0]]
y_test[idx[0]]
np.argmax(pred_y[idx[0]])
np.argmax(y_test[idx[0]])

#%%

# prob_y = model.predict(x_test)
# prob_y[0, :]
# plt.bar(np.arange(0, 10), prob_y[0, :], color='blue')

#%%
# K.backend.clear_session()
#%%

(_, y_train), (_, y_test) = mnist.load_data()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
pred = logreg.predict(x_test)
pred[:5]

#%%
