#%%
import os
print(os.getcwd())
os.chdir(r'D:\mayson\kmooc\ppt\week1')
print('current directory:', os.getcwd())
#%% mtcars

# pandas 모듈을 불러온다.
import pandas as pd 

# 데이터: https://www.kaggle.com/lavanya321/mtcars
# csv 파일 형식으로 되어있는 mtcars 데이터를 불러온다.
mtcars = pd.read_csv('mtcars.csv', encoding='cp949') 

# 앞부분의 일부 데이터를 출력
mtcars.head()

#%%

# mnist dataset을 제공하는 tensorflow 모듈을 불러온다.
import tensorflow as tf 

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 

# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pylab as plt 

# mnist dataset을 읽어온다.
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data() 

# 불러온 이미지 데이터(train_images)를 적절한 차원(28 x 28 pixel image)으로 변환한다.
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# 예시로 첫번째 이미지 데이터를 확인한다.
sample = np.squeeze(train_images[0], axis=-1) 

# grayscale의 colormap을 이용해 sample을 그린다.
plt.imshow(sample, cmap='gray') 

plt.close()

#%%

# jpg 형식의 image를 불러오기 위한 모듈을 불러온다. (RGB 채널 인식)
import matplotlib.image as mpimg 
# image를 쉽게 plot하기 위한 모듈을 불러온다.
from PIL import Image 
from copy import deepcopy

# 데이터: https://pixabay.com/ko/photos/%EA%B3%A0%EC%96%91%EC%9D%B4-%EA%B3%A0%EC%96%91%EC%9D%B4%EA%B3%BC%EC%9D%98-%ED%82%A4%ED%8B%B0-111793/
# 이미지를 그대로 읽어온다.
jpg_img = Image.open('./cat.jpg') 
jpg_img

# 행렬의 형태로 이미지를 읽어온다.
jpg_img_mat = mpimg.imread('./cat.jpg') 
jpg_img_mat
type(jpg_img_mat)
jpg_img_mat.shape

# Red -> Red이외의 pixel 값을 모두 0으로 지정
red_mat = deepcopy(jpg_img_mat)
red_mat[:, :, 1:3] = 0 
plt.imshow(red_mat)
red_mat[:,:,0]

# Green -> Green이외의 pixel 값을 모두 0으로 지정
green_mat = deepcopy(jpg_img_mat)
green_mat[:, :, [0, 2]] = 0 
plt.imshow(green_mat)
green_mat[:,:,1]

# Blue -> Blue이외의 pixel 값을 모두 0으로 지정
blue_mat = deepcopy(jpg_img_mat)
blue_mat[:, :, 0:2] = 0 
plt.imshow(blue_mat)
blue_mat[:,:,2]

#%%

# 행렬을 다루기 위한 모듈을 불러온다.
import numpy as np 
# plot을 그리기 위한 모듈을 불러온다.
import matplotlib.pylab as plt 
# wave format file을 다루기 위한 module을 불러온다.
from scipy.io import wavfile 

# 데이터: https://freesound.org/people/j1987/sounds/237050/
sample_rate, data = wavfile.read('curtain_sound.wav') # wave format file을 읽는다.

print(sample_rate)
print(data[:10])
# 2개의 channel로 구성되어 있다.
print(data.shape[1]) 

# 첫번째 channel만을 사용한다.
data = np.array([x[0] for x in data]) 

# 데이터 관측 시간을 계산한다.
times = np.arange(len(data))/float(sample_rate) 

# spectogram을 그리는 부분
plt.figure(figsize=(10, 3))
plt.fill_between(times, data) 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()

#%%

# 주어진 문장
text = '기계학습 방법론 을 이해 하여 인공 지능 을 만들어 봅시다 .'

# 띄어쓰기 단위로 문장을 토큰화(tokenize)
text_tokenized = text.split(' ') 

# 단어사전
vocab = {'.': 0,
         '기계학습': 1,
         '만들어': 2,
         '방법론': 3,
         '봅시다': 4,
         '을': 5,
         '이해': 6,
         '인공': 7,
         '지능': 8,
         '하여': 9}

# 문장의 각 토큰(token)들을 단어 사전에서 해당하는 번호로 치환하여 
# 단어들의 번호로 이루어진 순서열을 생성한다.
text_seqeunce = [vocab.get(x) for x in text_tokenized]
text_seqeunce

# 단어 사전의 크기, 즉 단어의 개수
vocab_size = len(vocab) 
vocab_size

import tensorflow as tf

# 단어들의 번호로 이루어진 순서열을 원-핫(one-hot) 벡터로 바꾼다.
one_hot = tf.keras.utils.to_categorical(text_seqeunce, 
                                        num_classes=vocab_size, 
                                        dtype='int')
one_hot

#%%




















