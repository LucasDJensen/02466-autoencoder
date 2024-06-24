import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shutil, sys
import joblib
import numpy as np

from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

from Autoencoder import Autoencoder
from load_data import load_data
from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH


# Filelists
# ==================================================
# eeg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
# eog_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
# emg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")

# eeg_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt") # "file containing the list of test EEG data")
# eog_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt") # "file containing the list of test EOG data")
# emg_test_data =os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt") # "file containing the list of test EMG data")

in_dir="C:/02466/02466-autoencoder/data/kornum_data/"

eeg_eval_data=os.path.join(in_dir, "eeg1/eval_list.txt")
eog_eval_data=os.path.join(in_dir, "eeg2/eval_list.txt")
emg_eval_data=os.path.join(in_dir, "emg/eval_list.txt")

# eeg_test_data=os.path.join(in_dir, "eeg1/test_list.txt")
# eog_test_data=os.path.join(in_dir, "eeg2/test_list.txt")
# emg_test_data =os.path.join(in_dir, "emg/test_list.txt")


# Configure
# ==================================================
artifact_detection = False
mask_artifacts = True

nclasses_model = 1
nclasses_data = 4 #2
frame_seq_len = 17
ndim = 129
sub_seq_len = 10
nsubseq = 8

if artifact_detection == True:
    artifacts_label = nclasses_data - 1
    nclasses_model = 1
    nclasses_data = 2
elif artifact_detection == False:
    if mask_artifacts == True:
        artifacts_label = nclasses_data - 1
        nclasses_model = nclasses_data - 1
    else:
        nclasses_model = nclasses_data
        artifacts_label = nclasses_data -1

# Load data and models
# ==================================================

val_data = load_data(eeg_filelist=os.path.abspath(eeg_eval_data),
                    eog_filelist=os.path.abspath(eog_eval_data),
                    emg_filelist=os.path.abspath(emg_eval_data),
                    data_shape_2=[frame_seq_len, ndim],
                    seq_len=sub_seq_len*nsubseq,
                    nclasses = nclasses_data, 
                    artifact_detection = artifact_detection,
                    artifacts_label = artifacts_label)

test_data = load_data(eeg_filelist=os.path.abspath(eeg_test_data),
                    eog_filelist=os.path.abspath(eog_test_data),
                    emg_filelist=os.path.abspath(emg_test_data),
                    data_shape_2=[frame_seq_len, ndim],
                    seq_len=sub_seq_len*nsubseq,
                    nclasses = 2, 
                    artifact_detection = True,
                    artifacts_label = (nclasses_data - 1))

# For 2D CNN Autoencoder
# xval = val_data.X2
# xtest = test_data.X2
# test_labels = test_data.y[:,1,0]
# autoencoder = tf.keras.models.load_model('image_ae.h5')
# encoder=tf.keras.Model(inputs=autoencoder.input,outputs=autoencoder.layers[7].output)
# serialized_tensor = tf.io.read_file('image_ae_features.tfrecord')
# val_features = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)
# gmm = joblib.load('image_ae_gmm.pkl')
# flat = False
# fxval = 0
# fxtest = 0
# data = "X2"
# cm_name = "image_ae_cm.png"

# For Dense Autoencoder
xval = val_data.X2
xtest = test_data.X2
test_labels = test_data.y[:,1,0]
fxval=tf.reshape(xval,(-1,17*129*3))
fxtest=tf.reshape(xtest,(-1,17*129*3))
autoencoder = tf.keras.models.load_model('dense_ae.h5')
encoder=tf.keras.Model(inputs=autoencoder.input,outputs=autoencoder.layers[4].output)
serialized_tensor = tf.io.read_file('dense_ae_features.tfrecord')
val_features = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)
gmm = joblib.load('dense_ae_gmm.pkl')
flat = True
data = "X2"
cm_name = "dense_ae_cm.png"

# For Temporal Autoencoder
# xval = val_data.X1
# xtest = test_data.X1
# test_labels = test_data.y[:,1,0] 
# autoencoder = tf.keras.models.load_model('ta.h5')
# encoder=tf.keras.Model(inputs=autoencoder.input,outputs=autoencoder.layers[8].output)
# serialized_tensor = tf.io.read_file('ta_features.tfrecord')
# val_features = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)
# gmm = joblib.load('ta_gmm.pkl')
# flat = False
# fxval = 0
# fxtest = 0
# data = "X1"
# cm_name = "ta_cm.png"

# Do reconstructions and calculate R(2) score
# ==================================================
if flat:
    val_reconstructions = autoencoder.predict(fxval)
    test_reconstructions = autoencoder.predict(fxtest)
    val_recon = tf.reshape(val_reconstructions,(-1,17,129,3))
    test_recon = tf.reshape(test_reconstructions,(-1,17,129,3))
else:
    val_recon = autoencoder.predict(xval)
    test_recon = autoencoder.predict(xtest)

r2_scores=[]
for c in range(3):
    r2_score_channel=[]
    for i in range(xval.shape[0]):
        if data == "X2":
            r2_score_channel.append(r2_score(tf.squeeze(xval[i,:,:,c]), tf.squeeze(val_recon[i,:,:,c])))
        elif data == "X1":
            r2_score_channel.append(r2_score(xval[i,:,c], val_recon[i,:,c]))
    r2_scores.append(tf.reduce_mean(r2_score_channel))

print("The average r2 score for validation set for each channel is: {}".format(r2_scores))
print("The average r2 score across the three channels for validation set is {}".format(tf.reduce_mean(r2_scores)))

r2_scores=[]
for c in range(3):
    r2_score_channel=[]
    for i in range(xtest.shape[0]):
        if data =="X2":
            r2_score_channel.append(r2_score(tf.squeeze(xtest[i,:,:,c]), tf.squeeze(test_recon[i,:,:,c])))
        elif data == "X1":
            r2_score_channel.append(r2_score(xval[i,:,c], test_recon[i,:,c]))
    r2_scores.append(tf.reduce_mean(r2_score_channel))

print("The average r2 score for test set for each channel is: {}".format(r2_scores))
print("The average r2 score across the three channels for test set is {}".format(tf.reduce_mean(r2_scores)))

# Evaluate GMM on test set
# ==================================================
test_features = encoder.predict(xtest)
mean_features = tf.reduce_mean(val_features,axis=0)
std_features = tf.math.reduce_std(val_features,axis=0)
test_embeddings = (test_features - mean_features)/std_features

log_likelihood = gmm.score_samples(test_embeddings)
test_scores = -log_likelihood

fpr_gmm, tpr_gmm, thresholds = roc_curve(test_labels, test_scores)

optimal_idx = np.argmax(tpr_gmm - fpr_gmm)
optimal_threshold = thresholds[optimal_idx]

# Define artifacts
artifacts = test_scores >= optimal_threshold
print("Number of artifacts detected:", len(artifacts))

# Confusion matrix
plt.figure()
cm = confusion_matrix(test_labels,artifacts)
sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=['non. art.','art.'],
            yticklabels=['non. art.','art.'],
            cmap="Blues")
plt.xlabel('Prediction',fontsize=13)
plt.ylabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.savefig(cm_name)
plt.show()
