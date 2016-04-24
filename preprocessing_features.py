import numpy as np
import pandas as pd
import warnings
from pandas import DataFrame, Series

video_number = 9
fall_starting_in_sec = 11
fall_ending_in_sec = 12
frame_per_sec = 2
seconds_after_fall=10

floats=[]
training_points=np.array(floats)
X_train=[]
y_train=[]

input_folder = 'files/'
video_file = input_folder + 'video-{}'.format(video_number)
path_txt = video_file +'.txt'
data  = np.genfromtxt(path_txt,delimiter=' ')
feature_vector_1 = np.array(data)
feature_vector_2 = np.array(data)

for x in range(len(data)):
        if x in range(fall_starting_in_sec*frame_per_sec,fall_ending_in_sec*frame_per_sec):
            y_train.append(1)
        else:
            y_train.append(0)

with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(len(data)):
            if(i==len(data)-1):
                feature_vector_2[i]=data[i]
                break
            try:
                after_fall_frames = data[i+1:i+1+frame_per_sec*seconds_after_fall]
            except IndexError:
                after_fall_frames = data[i+1:]
            feature_vector_2[i]=np.mean(after_fall_frames)

df = DataFrame({'f1' : feature_vector_1,'f2' : feature_vector_2, 'y': y_train})
df.to_csv(video_file+'_processed'+'.txt', header=None, index=None, sep=' ', mode='a')
