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
import pickle

test_video_number = 13
model_number = 14
best_threshold = -3.850000000000131

gmm_model = pickle.load(open('gmm_model{}.pickle'.format(model_number), 'rb'))
path = 'files/video-{}_processed.txt'.format(test_video_number)
data  = np.genfromtxt(path,delimiter=' ')
X_test = data[:,[0,1]]
y_test = np.reshape(data[:,2],(len(data),1))
log_likelihood = -gmm_model.score(X_test)
#print(log_likelihood)
print('SUMMARY EVALUATION:')
print('log_likelihood min: %.3f \nlog_likelihood.max: %.3f \nlog_likelihood mean: %.3f'
    % (log_likelihood.min(), log_likelihood.max(),log_likelihood.mean()))




y_predicted = [x < best_threshold for x in log_likelihood]
roc_score = roc_auc_score(y_test,y_predicted)
acc_score = accuracy_score(y_true=y_test, y_pred=y_predicted)
f1 = f1_score(y_true=y_test, y_pred=y_predicted)
mcc = matthews_corrcoef(y_true=y_test, y_pred=y_predicted)
precision = precision_score(y_true=y_test, y_pred=y_predicted)
recall = recall_score(y_true=y_test, y_pred=y_predicted)

print('\nAccuracy: %.3f \nError: %.3f'% (acc_score, 1-acc_score))
print()
print('Precision score: %.3f \nRecall score: %.3f \nF1 score:  %.3f' %(precision,recall,f1))
print()
print('AUC score: %.3f \nMCC score: %.3f' %(roc_score, mcc))

confmat = confusion_matrix(y_true=y_test, y_pred=y_predicted)
print('\nConfusion Matrix:')
print(confmat)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i,s=confmat[i, j],va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
plt.savefig('confusion-matrix')

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_predicted)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig('ROC-graph')
