#%%
import os
os.chdir(r'C:\Users\anseunghwan\Desktop\mayson\kmooc\ppt')
print('current directory:', os.getcwd())

#%%

# jpg 형식의 image를 불러오기 위한 모듈을 불러온다. (RGB 채널 인식)
import matplotlib.image as mpimg 
# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 
# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pylab as plt 

img_mat1 = np.rot90(mpimg.imread('./test_img1.jpg'), 3)
plt.imshow(img_mat1)

#%%

# cropping and reshape
# !pip install opencv-python
import cv2
img_cropped1 = img_mat1[1200:2000, 1250:1750, :]
img_resized1 = cv2.resize(img_cropped1, dsize=(28, 28))
img_resized1.shape
plt.imshow(img_resized1)

# model의 입력에 맞게 사진을 전처리
# 이미지를 회색조로 변환
def rgb2grayscale(img): 
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
img_gray1 = rgb2grayscale(img_resized1)
# scaling과 흑백반전
img_input1 = (255.0 - img_gray1) / 255.0
plt.imshow(img_input1, cmap='gray')

#%%

# 다른 이미지 예시

img_mat2 = np.rot90(mpimg.imread('./test_img2.jpg'), 3)
# cropping and reshape
img_cropped2 = img_mat2[1350:2000, 1100:1700, :]
img_resized2 = cv2.resize(img_cropped2, dsize=(28, 28))
# model의 입력에 맞게 사진을 전처리
# 이미지를 회색조로 변환
img_gray2 = rgb2grayscale(img_resized2)
# scaling과 흑백반전
img_input2 = (255.0 - img_gray2) / 255.0
plt.imshow(img_input2, cmap='gray')

#%%

'''introduction'''

# mnist dataset을 제공하는 tensorflow 모듈을 불러온다.
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
mnist = K.datasets.mnist

num_classes = 10
img_rows = img_cols = 28
input_shape = [img_rows, img_cols, 1]

#%%

'''linear model'''

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((len(x_train), img_rows*img_cols))
x_test = x_test.reshape((len(x_test), img_rows*img_cols))
x_train.shape
y_train = K.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = K.utils.to_categorical(y_test, num_classes=10, dtype='float32')
y_train[0]

#%%

input_layer = layers.Input((img_rows*img_cols))
dense_layer = layers.Dense(units=num_classes, activation='linear')
x = dense_layer(input_layer)

linear_model = K.models.Model(input_layer, x)
linear_model.summary()

linear_model.compile(optimizer='rmsprop',
                     loss='mse',
                     metrics=['mse'])
linear_model.fit(x_train, y_train,
                 batch_size=1024,
                 epochs=100,
                 validation_split=0.2)
    
#%%

pred_y1 = linear_model.predict(img_input1.reshape([1, img_rows*img_cols]))
plt.bar(np.arange(num_classes), np.squeeze(pred_y1), color='blue')
np.argmax(pred_y1)

pred_y2 = linear_model.predict(img_input2.reshape([1, img_rows*img_cols]))
plt.bar(np.arange(num_classes), np.squeeze(pred_y2), color='blue')
np.argmax(pred_y2)

#%%

'''glm'''

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((len(x_train), img_rows*img_cols))
x_test = x_test.reshape((len(x_test), img_rows*img_cols))
x_train.shape
y_train = K.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = K.utils.to_categorical(y_test, num_classes=10, dtype='float32')
y_train[0]

#%%

input_layer = layers.Input((img_rows*img_cols))
dense = layers.Dense(units=num_classes, activation='softmax',
                     kernel_regularizer=K.regularizers.l2(0.01))
x = dense(input_layer)

glm_model = K.models.Model(input_layer, x)
glm_model.summary()

glm_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
glm_model.fit(x_train, y_train,
              batch_size=1024,
              epochs=100,
              validation_split=0.2)

#%%

pred_y1 = glm_model.predict(img_input1.reshape([1, img_rows*img_cols]))
plt.bar(np.arange(num_classes), np.squeeze(pred_y1), color='blue')
np.argmax(pred_y1)

pred_y2 = glm_model.predict(img_input2.reshape([1, img_rows*img_cols]))
plt.bar(np.arange(num_classes), np.squeeze(pred_y2), color='blue')
np.argmax(pred_y2)

#%%

'''Random Forest'''

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((len(x_train), img_rows*img_cols))
x_test = x_test.reshape((len(x_test), img_rows*img_cols))
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train.shape
y_train[0]

#%%

from sklearn.ensemble import RandomForestClassifier 
rftree = RandomForestClassifier(random_state=1).fit(x_train, y_train)

#%%

pred_y1 = rftree.predict(img_input1.reshape([1, img_rows*img_cols]))
pred_y1

pred_y2 = rftree.predict(img_input2.reshape([1, img_rows*img_cols]))
pred_y2

#%%

'''CNN'''

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((len(x_train), img_rows, img_cols, 1))
x_test = x_test.reshape((len(x_test), img_rows, img_cols, 1))
x_train.shape
y_train = K.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = K.utils.to_categorical(y_test, num_classes=10, dtype='float32')
y_train[0]

#%%

input_layer = layers.Input(input_shape)
conv2d = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
x = conv2d(input_layer)
maxpool = layers.MaxPooling2D(pool_size=(2, 2))
x = maxpool(x)
dropout1 = layers.Dropout(rate=0.25)
x = dropout1(x)
flatten = layers.Flatten()
x = flatten(x)
dense = layers.Dense(units=128, activation='relu')
dropout2 = layers.Dropout(rate=0.5)
x = dropout2(x)
softmax = layers.Dense(units=num_classes, activation='softmax')
x = softmax(x)

cnn_model = K.models.Model(input_layer, x)
cnn_model.summary()

cnn_model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.fit(x_train, y_train,
              batch_size=1024,
              epochs=50,
              validation_split=0.2)

#%%

pred_y1 = cnn_model.predict(img_input1.reshape([1] + input_shape))
plt.bar(np.arange(num_classes), np.squeeze(pred_y1), color='blue')
np.argmax(pred_y1)

pred_y2 = cnn_model.predict(img_input2.reshape([1] + input_shape))
plt.bar(np.arange(num_classes), np.squeeze(pred_y2), color='blue')
np.argmax(pred_y2)

#%%

x_train[0][:10, :10, 0]
img_input2[:10, :10]

#%%

'''2D Fused Lasso'''

# 데이터 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((len(x_train), img_rows*img_cols))
x_test = x_test.reshape((len(x_test), img_rows*img_cols))
x_train.shape
y_train = K.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = K.utils.to_categorical(y_test, num_classes=10, dtype='float32')
y_train[0]

#%%

'''D matrix'''

D1 = (np.eye(img_rows*img_cols) + np.diag([-1]*(img_rows*img_cols-1), k=1))[:-1, :]
D1 = D1[[i for i in np.arange(D1.shape[0]) if not np.isin(i, np.arange(img_rows-1, D1.shape[0], img_rows))], :]
D1.shape
D2 = (np.eye(img_rows*img_cols) + np.diag([-1]*(img_rows*img_cols-img_rows), k=img_rows))[:-img_rows, :]
D2.shape
D = np.vstack((D1, D2))
D
D.shape

#%%

lambda_v = 0.1
# 사용자 정의 penalty function
def fusedlasso2d_regularizer(weights):
    return lambda_v * tf.reduce_sum(tf.math.abs(D @ weights))

#%%

input_layer = layers.Input((img_rows*img_cols))
dense = layers.Dense(units=num_classes, activation='softmax', use_bias=False,
                     kernel_regularizer=fusedlasso2d_regularizer)
x = dense(input_layer)

fusedlasso2d_model = K.models.Model(input_layer, x)
fusedlasso2d_model.summary()

fusedlasso2d_model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
fusedlasso2d_model.fit(x_train, y_train,
                       batch_size=1024,
                       epochs=100,
                       validation_split=0.2)

#%%

pred_y1 = fusedlasso2d_model.predict(img_input1.reshape([1, img_rows*img_cols]))
plt.bar(np.arange(num_classes), np.squeeze(pred_y1), color='blue')
np.argmax(pred_y1)

pred_y2 = fusedlasso2d_model.predict(img_input2.reshape([1, img_rows*img_cols]))
plt.bar(np.arange(num_classes), np.squeeze(pred_y2), color='blue')
np.argmax(pred_y2)

#%%
# K.backend.clear_session()
#%%
