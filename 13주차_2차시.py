#%%

'''모형의 구성과 모형 적합'''

# # mnist dataset을 제공하는 tensorflow 모듈을 불러온다.
# import tensorflow as tf
# import tensorflow.keras as K
# from tensorflow.keras import layers
# # 행렬을 다루기 위한 모듈을 불러온다.
# import numpy as np 

#%%

input_layer = layers.Input((784))
dense_layer = layers.Dense(units=20, activation='relu')
x = dense_layer(input_layer)
softmax_layer = layers.Dense(units=10, activation='softmax')
x = softmax_layer(x)

model = K.models.Model(input_layer, x)
model.summary()

#%%

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#%%

model.fit(x_train, y_train,
          batch_size=128,
          epochs=30,
          validation_split=0.2)

#%%

pred_y_prob = model.predict(x_test)

#%%

input_layer = layers.Input((784))
dense_layer = layers.Dense(units=100, activation='relu')
x = dense_layer(input_layer)
softmax_layer = layers.Dense(units=10, activation='softmax')
x = softmax_layer(x)

model = K.models.Model(input_layer, x)
model.summary()


input_layer = layers.Input((784))
dense_layer1 = layers.Dense(units=100, activation='relu')
x = dense_layer1(input_layer)
dense_layer2 = layers.Dense(units=20, activation='relu')
x = dense_layer2(x)
softmax_layer = layers.Dense(units=10, activation='softmax')
x = softmax_layer(x)

model = K.models.Model(input_layer, x)
model.summary()

#%%

input_layer = layers.Input((784))
dense_layer = layers.Dense(units=20, activation='relu')
x = dense_layer(input_layer)
dropout = layers.Dropout(rate=0.3)
x = dropout(x)
softmax_layer = layers.Dense(units=10, activation='softmax')
x = softmax_layer(x)

model = K.models.Model(input_layer, x)
model.summary()

#%%

input_layer = layers.Input((784))
dense_layer1 = layers.Dense(units=100, activation='relu',
                            kernel_regularizer=K.regularizers.l1(0.01))
x = dense_layer1(input_layer)
dense_layer2 = layers.Dense(units=20, activation='relu',
                            kernel_regularizer=K.regularizers.l2(0.1))
x = dense_layer2(x)
softmax_layer = layers.Dense(units=10, activation='softmax')
x = softmax_layer(x)

model = K.models.Model(input_layer, x)
model.summary()

#%%

'''CNN'''

# 고양이 그림

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 
# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pylab as plt 
# jpg 형식의 image를 불러오기 위한 모듈을 불러온다. (RGB 채널 인식)
import matplotlib.image as mpimg 
# image를 쉽게 plot하기 위한 모듈을 불러온다.
from PIL import Image 
from scipy import signal

# 이미지를 그대로 읽어온다.
# 데이터: https://pixabay.com/ko/photos/%EA%B3%A0%EC%96%91%EC%9D%B4-%EA%B3%A0%EC%96%91%EC%9D%B4%EA%B3%BC%EC%9D%98-%ED%82%A4%ED%8B%B0-111793/
jpg_img = Image.open('./cat.jpg') 
jpg_img

c1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
jpg_img_mat = mpimg.imread('./cat.jpg') 
edge1 = signal.convolve2d(jpg_img_mat[:, :, 0], c1, 'valid')[..., np.newaxis]
for i in range(1, 3):
    edge1 = np.concatenate((edge1, signal.convolve2d(jpg_img_mat[:, :, i], c1, 'valid')[..., np.newaxis]), axis=-1)    
plt.imshow(edge1)

c2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
jpg_img_mat = mpimg.imread('./cat.jpg') 
edge2 = signal.convolve2d(jpg_img_mat[:, :, 0], c2, 'valid')[..., np.newaxis]
for i in range(1, 3):
    edge2 = np.concatenate((edge2, signal.convolve2d(jpg_img_mat[:, :, i], c2, 'valid')[..., np.newaxis]), axis=-1)    
plt.imshow(edge2)

c3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
jpg_img_mat = mpimg.imread('./cat.jpg') 
edge3 = signal.convolve2d(jpg_img_mat[:, :, 0], c3, 'valid')[..., np.newaxis]
for i in range(1, 3):
    edge3 = np.concatenate((edge3, signal.convolve2d(jpg_img_mat[:, :, i], c3, 'valid')[..., np.newaxis]), axis=-1)    
plt.imshow(edge3)


#%%

batch_size = 128
num_classes = 10
epochs = 10

# mnist dataset을 제공하는 tensorflow 모듈을 불러온다.
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 

# mnist dataset을 읽어온다.
mnist = K.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%

img_rows = img_cols = 28
x_train = x_train.reshape((len(x_train), img_rows, img_cols, 1))
x_test = x_test.reshape((len(x_test), img_rows, img_cols, 1))
input_shape = (img_rows, img_cols, 1)
x_train.shape
x_test.shape

#%%

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = K.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_train[0]
y_test = K.utils.to_categorical(y_test, num_classes=10, dtype='float32')

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
# K.backend.clear_session()
#%%
