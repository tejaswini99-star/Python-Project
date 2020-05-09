import pandas as pd
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd  # To play sound in the notebook
import os
import librosa
from playsound import playsound



#load the data
df = pd.read_csv("speakers_all.csv", header=0)

# Check the data of speakers_all data set
print(df.shape, 'is the shape of the dataset')
print('------------------------')
df.drop(df.columns[8:12],axis = 1, inplace = True)
print(df.columns)
print(df.describe())
print(df.groupby("native_language")['age'].describe().sort_values(by=['count'],ascending=False))
# Check the data of recordings data set
files =  os.listdir('recordings')
print(files)
print (len([name for name in os.listdir('recordings') if os.path.isfile(os.path.join('recordings', name))]))
print(df.groupby("filename")['age'].describe().sort_values(by=['count'],ascending=False).head(10))
fname1 = 'recordings/luo1.mp3'
playsound(fname1)
