# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1y2NWeKCCVlXLvhIWDT1MsEGB9ZkUHzWO
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import IPython.display as ipd
import librosa
import librosa.display

#load the data
df = pd.read_csv("/content/speakers_all.csv", header=0)

# Check the data of speakers_all data set
print(df.shape, 'is the shape of the dataset')
print('------------------------')
df.drop(df.columns[8:12],axis = 1, inplace = True)
print(df.columns)
print(df.describe())
print(df.groupby("native_language")['age'].describe().sort_values(by=['count'],ascending=False))
# Check the data of recordings data set
files=[]
files =  os.listdir('/content/drive/My Drive/Recordings/recordings')
print(files)
print (len([name for name in os.listdir('/content/drive/My Drive/Recordings/recordings') if os.path.isfile(os.path.join('recordings', name))]))
print(df.groupby("filename")['age'].describe().sort_values(by=['count'],ascending=False).head(10))
for i in range(0,2138):
  fname1 =os.listdir('/content/drive/My Drive/Recordings/recordings')[i]
  print(fname1)


x, sr = librosa.load('/content/drive/My Drive/Recordings/recordings/afrikaans1.wav')
print(x.shape)
print(sr)
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
plt.show()
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.show()


ipd.Audio('/content/drive/My Drive/Recordings/recordings/afrikaans1.wav')

import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D
from keras.models import Sequential
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import pandas as pd
import random
import os

import warnings
warnings.filterwarnings('ignore')
a = os.listdir('/content/drive/My Drive/Recordings/recordings')
# Read Data
data = pd.read_csv('/speakers_all.csv')
data.head(5)
print(data.shape)
data =data[32:]
valid_data = data[['age', 'birthplace','filename' ,'native_language','sex', 'speakerid','country']]
gender = {'male': 1,'female': 2}
valid_data.sex = [gender[item] for item in data.sex]

print(valid_data.shape)

y, sr = librosa.load('/content/drive/My Drive/Recordings/recordings/afrikaans1.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
print(ps.shape)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()

valid_data['path'] = '/' + valid_data['filename'].astype('str')

D = [] # Dataset

for row in valid_data.itertuples():
      y, sr = librosa.load('/content/drive/My Drive/Recordings/recordings' + row.path+'.wav', duration=2.97)
      ps = librosa.feature.melspectrogram(y=y, sr=sr)
      if ps.shape != (128, 128): continue
      D.append( (ps, row.sex) )
      print("Number of samples: ", len(D))
dataset = D
random.shuffle(dataset)



train = dataset[:2000]
test =  dataset[2000:]


X_train, y_train = zip(*train)
X_test, y_test = zip(*test)
# Reshape for CNN input
X_train = np.array([x.reshape( (128, 128, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 128, 1) ) for x in X_test])

# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train))
y_test = np.array(keras.utils.to_categorical(y_test))

num_classes = y_test.shape[1]
input_shape=(128, 128, 1)
model = Sequential()
model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape,padding='same',activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(48, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
	optimizer="Adam",
	loss="categorical_crossentropy",
	metrics=['accuracy'])

model.fit(
	x=X_train,
	y=y_train,
    epochs=3,
    batch_size=128,
    validation_data= (X_test, y_test))

score = model.evaluate(x=X_test,y=y_test)


print('Test accuracy:', score[1])

SAMPLE_RATE = 44100
fname_f = '/content/drive/My Drive/Recordings/recordings/french1.wav' 
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

plt.figure(figsize=(10, 4))
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Audio sampled at 44100 hrz')

SAMPLE_RATE = 6000
fname_f = '/content/drive/My Drive/Recordings/recordings/french1.wav' 
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

plt.figure(figsize=(10, 4))
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Audio sampled at 6000 hrz')

SAMPLE_RATE = 1000
fname_f = '/content/drive/My Drive/Recordings/recordings/french1.wav' 
y, sr = librosa.core.load(fname_f, sr=SAMPLE_RATE, duration = 5)

plt.figure(figsize=(10, 4))
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Audio sampled at 1000 hrz')

SAMPLE_RATE = 44100
fname_f = '/content/drive/My Drive/Recordings/recordings/french2.wav' 
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

plt.figure(figsize=(10, 4))
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Audio sampled at 44100 hrz')

SAMPLE_RATE = 6000
fname_f = '/content/drive/My Drive/Recordings/recordings/french2.wav' 
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

plt.figure(figsize=(10, 4))
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Audio sampled at 6000 hrz')

SAMPLE_RATE = 1000
fname_f = '/content/drive/My Drive/Recordings/recordings/french2.wav' 
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5)

plt.figure(figsize=(10, 4))
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Audio sampled at 1000 hrz')

SAMPLE_RATE = 22050
fname_f = '/content/drive/My Drive/Recordings/recordings/french1.wav'  
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5) # Chop audio at 5 secs... 
mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc = 5) # 5 MFCC components

plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()

SAMPLE_RATE = 22050
fname_m = '/content/drive/My Drive/Recordings/recordings/french2.wav'  
y, sr = librosa.load(fname_m, sr=SAMPLE_RATE, duration = 5)
mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc = 5)

plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
librosa.display.specshow(mfcc)
plt.ylabel('MFCC')
plt.colorbar()

SAMPLE_RATE = 22050
fname_f = '/content/drive/My Drive/Recordings/recordings/french1.wav'  
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5) # Chop audio at 5 secs... 
melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.amplitude_to_db(melspec)

# Display the log mel spectrogram
plt.figure(figsize=(10,8))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Log mel spectrogram for female')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()

SAMPLE_RATE = 22050
fname_m = '/content/drive/My Drive/Recordings/recordings/french2.wav'  
y, sr = librosa.load(fname_m, sr=SAMPLE_RATE, duration = 5) # Chop audio at 5 secs... 
melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.amplitude_to_db(melspec)

# Display the log mel spectrogram
plt.figure(figsize=(10,8))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Log mel spectrogram for male')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()

SAMPLE_RATE = 22050
fname_f = '/content/drive/My Drive/Recordings/recordings/english385.wav'  
y, sr = librosa.load(fname_f, sr=SAMPLE_RATE, duration = 5) 
y_harmonic, y_percussive = librosa.effects.hpss(y)

ipd.Audio(y_harmonic, rate=sr)

ipd.Audio(y_percussive, rate=sr)