import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.mixture import GMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
import pickle

#Parameters
starting_video_number = 1
number_of_training_videos = 14
frame_per_sec = 2

stdsc = StandardScaler()
gmm = mixture.GMM(n_components=3)
best_thresholds = []
acc_scores = []
roc_scores = []
mcc_scores = []
mses = []
X_train = np.empty([1, 2])
gmm_models_numbers = number_of_training_videos+1

print('Training GMM ..')
for kfold in range(starting_video_number,starting_video_number+number_of_training_videos):
    for i in range(1,number_of_training_videos):
        if i == kfold-i+number_of_training_videos:
            break
        path = 'files/video-{}_processed.txt'.format(i)
        data  = np.genfromtxt(path,delimiter=' ')
        X_train = np.concatenate((X_train,data[:,[0,1]]))
    X_train = stdsc.fit_transform(X_train)
    gmm.fit(X_train)

    test_fold = number_of_training_videos-kfold+1
    path = 'files/video-{}_processed.txt'.format(test_fold)
    data  = np.genfromtxt(path,delimiter=' ')
    X_test = data[:,[0,1]]
    y_test = np.reshape(data[:,2],(len(data),1))
    log_likelihood = -gmm.score(X_test)

    thresholds = np.arange(-10,0,0.01)
    max_mcc_score=0
    best_threshold=0
    for threshold in thresholds:

        # gradients=np.diff(X_test[:,0])
        # maxima_num=0
        # max_locations=[]
        # count=0
        # for i in gradients[:-1]:
        #     count+=1
        #     if ((cmp(i,0)>0) & (cmp(gradients[count],0)<0) & (i != gradients[count])):
        #     maxima_num+=1
        #     max_locations.append(count)
        # X_test[max_locations[]]

        y_predicted = [x < threshold for x in log_likelihood]
        mcc = matthews_corrcoef(y_true=y_test, y_pred=y_predicted)
        if  mcc > max_mcc_score :
            max_mcc_score = mcc
            best_threshold = threshold
    best_thresholds.append(best_threshold)
    print('Best Thresold for GMM{}: {}'.format(kfold, best_threshold))

    y_predicted = [x < best_threshold for x in log_likelihood]
    roc_score = roc_auc_score(y_test,y_predicted)
    mcc_score = matthews_corrcoef(y_true=y_test, y_pred=y_predicted)
    acc_score = accuracy_score(y_true=y_test, y_pred=y_predicted)
    #acc_scores.append(acc_score)
    #roc_scores.append(roc_score)
    mcc_scores.append(mcc_score)
    mse = mean_squared_error(y_test, y_predicted)
    mses.append(mse)
    print('Fold %s: ROC score: %.3f, Accuracy: %.3f, MCC: %.3f' % (kfold,roc_score,acc_score,mcc_score))
    print()
    pickle.dump(gmm, open('gmm_model{}.pickle'.format(kfold), 'wb'))



m = max(mcc_scores)
m_index = [i for i, j in enumerate(mcc_scores) if j == m]
gmm_threshold = best_thresholds[m_index[0]]

with open('gmm-pram.txt', 'a') as the_file:
    the_file.write(str(m_index[0]))
    the_file.write('\n')
    the_file.write(str(gmm_threshold))

for i in range(1,gmm_models_numbers):
    if i != m_index[0]:
        os.remove('gmm_model{}.pickle'.format(i))
print('LOOCV mean MSE: {}'.format(np.mean(mses)))
print('GMM selected: GMM {}'.format(m_index[0]))
print('GMM saved.')
