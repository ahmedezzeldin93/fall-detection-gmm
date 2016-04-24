import numpy as np
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
import pickle

#Parameters
starting_video_number = 1
number_of_training_videos = 14
frame_per_sec = 2


stdsc = StandardScaler()
gmm = mixture.GMM(n_components=3)
print('Training GMM ..')
X_train = np.empty([1, 2])
for i in range(starting_video_number,starting_video_number+number_of_training_videos):
    path = 'files/video-{}_processed.txt'.format(i)
    data = np.genfromtxt(path,delimiter=' ')
    X_train = np.concatenate((X_train,data[:,[0,1]]))
gmm.fit(X_train)

pickle.dump(gmm, open('gmm_model.pickle', 'wb'))
print('GMM saved.')
