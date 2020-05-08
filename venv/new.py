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
a = os.listdir('recordings')
# Read Data
data = pd.read_csv('speakers_all.csv')
data.head(5)
print(data.shape)
data =data[32:]
print(data['country'].isnull().sum())
data=data.fillna("na")
valid_data = data[['age', 'birthplace','filename' ,'native_language','sex', 'speakerid','country']]
desh = {'senegal':  1,
'cameroon': 2,
'nigeria' : 3,
'haiti':  4,
'usa':  5,
'jamaica':  6,
'liberia':  7,
'nicaragua':  8,
'south africa': 9,
'india':  10,
'sri lanka':  11,
'switzerland' :12,
'timor-leste':  13,
'papua new guinea' :14,
'ivory coast':  15,
'ghana':  16,
'kosovo': 17,
'albania':  18,
'morocco':  19,
'ethiopia': 20,
'saudi arabia': 21,
'egypt':  22,
'lebanon':  23,
'qatar':  24,
'tunisia':  25,
'iraq': 26,
'jordan': 27,
'kuwait': 28,
'syria':  29,
'united arab emirates': 30,
'israel (occupied territory)':  31,
'israel': 32,
'algeria':33,
'yemen':  34,
'bahrain':  35,
'libya':  36,
'oman': 37,
'uk': 38,
'sudan':  39,
'armenia':  40,
'iran': 41,
'republic of georgia':  42,
'russia': 43,
'azerbaijan': 44,
'guinea': 45,
'china':  46,
'mali': 47,
'spain':  48,
'germany':  49,
'belarus':  50,
'bangladesh': 51,
'zambia': 52,
'bosnia and herzegovina': 53,
'bulgaria': 54,
'myanmar':  55,
'northern mariana islands': 56,
'chile':  57,
'philippines':  58,
'malawi': 59,
'croatia':  60,
'czech republic': 61,
'denmark':  62,
'afghanistan':  63,
'netherlands':  64,
'belgium':  65,
'south korea':  66,
'canada': 67,
'us virgin islands':  68,
'malaysia': 69,
'australia':  70,
'ireland':  71,
'guyana': 72,
'fiji': 73,
'antigua and barbuda':  74,
'panama': 75,
'barbados': 76,
'new zealand':  77,
'singapore':  78,
'isle of man':  79,
'belize': 80,
'bolivia':  81,
'trinidad': 82,
'virginia': 83,
'the bahamas':  84,
'italy':  85,
'pakistan': 86,
'estonia':  87,
'equatorial guinea':  88,
'faroe islands':  89,
'finland':  90,
'france': 91,
'democratic republic of congo': 92,
'andorra':  93,
'portugal': 94,
'gabon':  95,
'burkina faso': 96,
'martinique': 97,
'uganda': 98,
'honduras': 99,
'austria':  100,
'liechtenstein':  101,
'greece': 102,
'cyprus': 103,
'kenya':  104,
'taiwan': 105,
'niger':  106,
'laos': 107,
'hungary':  108,
'romania':  109,
'iceland':  110,
'togo': 111,
'indonesia':  112,
'japan':  113,
'botswana': 114,
'kazakhstan': 115,
'cambodia': 116,
'angola': 117,
'kyrgyzstan': 118,
'tanzania': 119,
'sierra leone': 120,
'federated states of micronesia': 121,
'latvia': 122,
'lithuania':  123,
'luxembourg': 124,
'macedonia':  125,
'madagascar': 126,
'malta':  127,
'mauritius':  128,
'mongolia': 129,
'namibia':  130,
'nepal':  131,
'norway': 132,
'curacao':  133,
'poland': 134,
'brazil': 135,
'mexico': 136,
'peru': 137,
'moldova':  138,
'romanian': 139,
'ukraine':  140,
'burundi':  141,
'uzbekistan': 142,
'rwanda': 143,
'solomon islands':  144,
'chad': 145,
'bosnia': 146,
'serbia': 147,
'montenegro': 148,
'yugoslavia': 149,
'lesotho':  150,
'zimbabwe': 151,
'sicily': 152,
'slovak republic':  153,
'slovakia': 154,
'slovenia': 155,
'somalia':  156,
'venezuela':  157,
'el salvador':  158,
'colombia': 159,
'cuba': 160,
'ecuador':  161,
'puerto rico':  162,
'guatemala':  163,
'uruguay':  164,
'argentina':  165,
'costa rica': 166,
'dominican republic': 167,
'sweden': 168,
'tajikistan': 169,
'thailand': 170,
'tibet':  171,
'eritrea':  172,
'turkey': 173,
'turkmenistan': 174,
'vietnam':  175,
'benin':  176,
'na':177
}
valid_data.country = [desh[item] for item in data.country]
print(valid_data)
y, sr = librosa.load('recordings/afrikaans1.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
print(ps.shape)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()

valid_data['path'] = '/' + valid_data['filename'].astype('str')

D = [] # Dataset

for row in valid_data.itertuples():
      y, sr = librosa.load('recordings' + row.path+'.wav', duration=2.97)
      ps = librosa.feature.melspectrogram(y=y, sr=sr)
      if ps.shape != (128, 128): continue
      D.append( (ps, row.country) )
      print("Number of samples: ", len(D))
dataset = D
random.shuffle(dataset)



train = dataset[:50]
test =  dataset[50:100]


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
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(
  optimizer="Adam",
  loss="categorical_crossentropy",
  metrics=['accuracy'])

model.fit(
  x=X_train,
  y=y_train,
    epochs=3,
    batch_size=128,
    validation_data= (X_test, y_test[0]))

score = model.evaluate(x=X_test,y=y_test[0])


print('Test accuracy:', score[1])

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
data = pd.read_csv('/content/drive/My Drive/Recordings/speakers_all.csv')
data.head(5)
print(data.shape)
data =data[32:]
print(data['country'].isnull().sum())
data=data.fillna("na")
valid_data = data[['age', 'birthplace','filename' ,'native_language','sex', 'speakerid','country']]
desh = {'senegal':  1,
'cameroon': 2,
'nigeria' : 3,
'haiti':  4,
'usa':  5,
'jamaica':  6,
'liberia':  7,
'nicaragua':  8,
'south africa': 9,
'india':  10,
'sri lanka':  11,
'switzerland' :12,
'timor-leste':  13,
'papua new guinea' :14,
'ivory coast':  15,
'ghana':  16,
'kosovo': 17,
'albania':  18,
'morocco':  19,
'ethiopia': 20,
'saudi arabia': 21,
'egypt':  22,
'lebanon':  23,
'qatar':  24,
'tunisia':  25,
'iraq': 26,
'jordan': 27,
'kuwait': 28,
'syria':  29,
'united arab emirates': 30,
'israel (occupied territory)':  31,
'israel': 32,
'algeria':33,
'yemen':  34,
'bahrain':  35,
'libya':  36,
'oman': 37,
'uk': 38,
'sudan':  39,
'armenia':  40,
'iran': 41,
'republic of georgia':  42,
'russia': 43,
'azerbaijan': 44,
'guinea': 45,
'china':  46,
'mali': 47,
'spain':  48,
'germany':  49,
'belarus':  50,
'bangladesh': 51,
'zambia': 52,
'bosnia and herzegovina': 53,
'bulgaria': 54,
'myanmar':  55,
'northern mariana islands': 56,
'chile':  57,
'philippines':  58,
'malawi': 59,
'croatia':  60,
'czech republic': 61,
'denmark':  62,
'afghanistan':  63,
'netherlands':  64,
'belgium':  65,
'south korea':  66,
'canada': 67,
'us virgin islands':  68,
'malaysia': 69,
'australia':  70,
'ireland':  71,
'guyana': 72,
'fiji': 73,
'antigua and barbuda':  74,
'panama': 75,
'barbados': 76,
'new zealand':  77,
'singapore':  78,
'isle of man':  79,
'belize': 80,
'bolivia':  81,
'trinidad': 82,
'virginia': 83,
'the bahamas':  84,
'italy':  85,
'pakistan': 86,
'estonia':  87,
'equatorial guinea':  88,
'faroe islands':  89,
'finland':  90,
'france': 91,
'democratic republic of congo': 92,
'andorra':  93,
'portugal': 94,
'gabon':  95,
'burkina faso': 96,
'martinique': 97,
'uganda': 98,
'honduras': 99,
'austria':  100,
'liechtenstein':  101,
'greece': 102,
'cyprus': 103,
'kenya':  104,
'taiwan': 105,
'niger':  106,
'laos': 107,
'hungary':  108,
'romania':  109,
'iceland':  110,
'togo': 111,
'indonesia':  112,
'japan':  113,
'botswana': 114,
'kazakhstan': 115,
'cambodia': 116,
'angola': 117,
'kyrgyzstan': 118,
'tanzania': 119,
'sierra leone': 120,
'federated states of micronesia': 121,
'latvia': 122,
'lithuania':  123,
'luxembourg': 124,
'macedonia':  125,
'madagascar': 126,
'malta':  127,
'mauritius':  128,
'mongolia': 129,
'namibia':  130,
'nepal':  131,
'norway': 132,
'curacao':  133,
'poland': 134,
'brazil': 135,
'mexico': 136,
'peru': 137,
'moldova':  138,
'romanian': 139,
'ukraine':  140,
'burundi':  141,
'uzbekistan': 142,
'rwanda': 143,
'solomon islands':  144,
'chad': 145,
'bosnia': 146,
'serbia': 147,
'montenegro': 148,
'yugoslavia': 149,
'lesotho':  150,
'zimbabwe': 151,
'sicily': 152,
'slovak republic':  153,
'slovakia': 154,
'slovenia': 155,
'somalia':  156,
'venezuela':  157,
'el salvador':  158,
'colombia': 159,
'cuba': 160,
'ecuador':  161,
'puerto rico':  162,
'guatemala':  163,
'uruguay':  164,
'argentina':  165,
'costa rica': 166,
'dominican republic': 167,
'sweden': 168,
'tajikistan': 169,
'thailand': 170,
'tibet':  171,
'eritrea':  172,
'turkey': 173,
'turkmenistan': 174,
'vietnam':  175,
'benin':  176,
'na':177
}
valid_data.country = [desh[item] for item in data.country]
print(valid_data)
y, sr = librosa.load('/content/drive/My Drive/Recordings/recordings/afrikaans1.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
print(ps.shape)
librosa.display.specshow(ps, y_axis='mel', x_axis='time')
plt.show()

valid_data['path'] = '/' + valid_data['filename'].astype('str')

D = [] # Dataset

for row in valid_data.itertuples():
      y, sr = librosa.load('recordings' + row.path+'.wav', duration=2.97)
      ps = librosa.feature.melspectrogram(y=y, sr=sr)
      if ps.shape != (128, 128): continue
      D.append( (ps, row.country) )
      print("Number of samples: ", len(D))
dataset = D
random.shuffle(dataset)



train = dataset[:50]
test =  dataset[50:100]


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
input_shape=(128, 128, 1)

model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))

model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(
  optimizer="Adam",
  loss="categorical_crossentropy",
  metrics=['accuracy'])

model.fit(
  x=X_train,
  y=y_train,
    epochs=3,
    batch_size=128,
    validation_data= (X_test, y_test[0]))

score = model.evaluate(x=X_test,y=y_test[0])


print('Test accuracy:', score[1])