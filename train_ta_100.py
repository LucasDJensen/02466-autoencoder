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

from Autoencoder import TemporalAutoencoder
from load_data import load_data
from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH


# Filelists
# ==================================================
eeg_train_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/train_list.txt") #file containing the list of training EEG data
eog_train_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/train_list.txt") #file containing the list of training EOG data")
emg_train_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/train_list.txt") #"file containing the list of training EMG data")

eog_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
eeg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
emg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")

eeg_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt") # "file containing the list of test EEG data")
eog_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt") # "file containing the list of test EOG data")
emg_test_data =os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt") # "file containing the list of test EMG data")

# in_dir="C:/02466/02466-autoencoder/data/kornum_data/"
# eeg_train_data=os.path.join(in_dir, "eeg1/train_list.txt")
# eog_train_data=os.path.join(in_dir, "eeg2/train_list.txt")
# emg_train_data=os.path.join(in_dir, "emg/train_list.txt")

# eeg_eval_data=os.path.join(in_dir, "eeg1/eval_list.txt")
# eog_eval_data=os.path.join(in_dir, "eeg2/eval_list.txt")
# emg_eval_data=os.path.join(in_dir, "emg/eval_list.txt")

# eeg_test_data=os.path.join(in_dir, "eeg1/test_list.txt")
# eog_test_data=os.path.join(in_dir, "eeg2/test_list.txt")
# emg_test_data =os.path.join(in_dir, "emg/test_list.txt")

# Configure
# ==================================================
learning_rate = 0.0001
number_of_epochs = 100
batch_size = 32

artifact_detection = False
mask_artifacts = True

nclasses_model = 1
nclasses_data = 4 #2
frame_seq_len = 17
ndim = 129
sub_seq_len = 10
nsubseq = 8

latent_dim = 100
GMM_components = 15

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

# Load data
# ==================================================
train_data = load_data( eeg_filelist=os.path.abspath(eeg_train_data),
                        eog_filelist=os.path.abspath(eog_train_data),
                        emg_filelist=os.path.abspath(emg_train_data),
                        data_shape_2=[frame_seq_len, ndim],
                        seq_len=sub_seq_len*nsubseq,
                        nclasses = nclasses_data, 
                        artifact_detection = artifact_detection,
                        artifacts_label = artifacts_label)

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


#xtrain = train_data.X1
xval = val_data.X1
xtest = test_data.X1
test_labels = test_data.y[:,1,0] 

input_shape = xtrain.shape[1:]
print("Shape of training tensor: {}".format(xtrain.shape))

# Plot example eeg
# ==================================================
sampling_rate = 128  # Hz
n_samples = len(xval[0,:,0])
time_vector = tf.range(n_samples) / sampling_rate

plt.figure(figsize=(10, 5))
plt.plot(time_vector, xtrain[0,:,0], label='EEG1 Signal',linewidth=0.75)
plt.plot(time_vector, xtrain[0,:,1], label='EEG2 Signal',linewidth=0.75)
plt.plot(time_vector, xtrain[0,:,2], label='EMG Signal',linewidth=0.75)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('EEG Signal')
plt.legend()
plt.grid(True)
plt.savefig("sample_eeg.png")


# Define and train autoencoder
# ==================================================
model = TemporalAutoencoder(input_shape, activation_function = "elu", latent_dim=latent_dim, learning_rate=learning_rate)
autoencoder = model.autoencoder()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
fitted_model=autoencoder.fit(xtrain, xtrain, epochs=number_of_epochs, batch_size=batch_size,
                        validation_data=(xval,xval), callbacks=[early_stopping])

autoencoder.save('ta_100.h5')
#autoencoder = tf.keras.models.load_model('ta_100.h5')

# Create reconstructions and plot one
# ==================================================
val_recon = autoencoder.predict(xval)
test_recon = autoencoder.predict(xtest)

"""
plt.figure(figsize=(10, 5))
plt.plot(time_vector, xval[0,:,0], label='EEG1 Signal',linewidth=0.75)
plt.plot(time_vector, xval[0,:,1], label='EEG2 Signal',linewidth=0.75)
plt.plot(time_vector, xval[0,:,2], label='EMG Signal',linewidth=0.75)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('EEG Signal')
plt.legend()
plt.grid(True)
plt.savefig("validation_signal.png")
"""
plt.figure(figsize=(10, 5))
plt.plot(time_vector, val_recon[0,:,0], label='EEG1 Signal',linewidth=0.75)
plt.plot(time_vector, val_recon[0,:,1], label='EEG2 Signal',linewidth=0.75)
plt.plot(time_vector, val_recon[0,:,2], label='EMG Signal',linewidth=0.75)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('EEG Signal')
plt.legend()
plt.grid(True)
plt.savefig("validation_signal_reconstructed_100.png")


# Calculate R(2) for each epoch and the average over the channels
# ==================================================
r2_scores=[]
for c in range(3):
    r2_score_channel=[]
    for i in range(xval.shape[0]):
        r2_score_channel.append(r2_score(xval[i,:,c], val_recon[i,:,c]))
    r2_scores.append(tf.reduce_mean(r2_score_channel))

print("The average r2 score for each channel is: {}".format(r2_scores))
print("The average r2 score across the three channels is {}".format(tf.reduce_mean(r2_scores)))

r2_scores=[]
for c in range(3):
    r2_score_channel=[]
    for i in range(xtest.shape[0]):
        r2_score_channel.append(r2_score(xtest[i,:,c], test_recon[i,:,c]))
    r2_scores.append(tf.reduce_mean(r2_score_channel))

print("The average r2 score for test set for each channel is: {}".format(r2_scores))
print("The average r2 score across the three channels for test set is {}".format(tf.reduce_mean(r2_scores)))


# Plot training and validation loss
# ==================================================
train_loss = fitted_model.history['loss']
val_loss = fitted_model.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.savefig("train_val_loss_100.png")

# Extract embedding vectors
# ==================================================
#autoencoder=tf.keras.models.load_model('ta.h5')
encoder=tf.keras.Model(inputs=autoencoder.input,outputs=autoencoder.layers[8].output)

val_features = encoder.predict(xval)
test_features = encoder.predict(xtest)

mean_features = tf.reduce_mean(val_features,axis=0)
std_features = tf.math.reduce_std(val_features,axis=0)

val_embeddings = (val_features - mean_features)/std_features
test_embeddings = (test_features - mean_features)/std_features

print("Shape of embedding vectors:", val_embeddings.shape)

# Save validation feature vectors
serialized_tensor = tf.io.serialize_tensor(val_features)
tf.io.write_file('ta_100_features.tfrecord', serialized_tensor)

# Do PCA on validation embeddings
# ==================================================
pca = PCA()
pca.fit(val_embeddings)
projected_data = pca.transform(val_embeddings)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Explained Variance')
plt.grid()
plt.tight_layout()
plt.savefig("ta_100_PCA.png")

# Fit GMM with n_components = 15 
# ==================================================
cov_type = 'diag'
initialization_method = 'random'
gmm = GaussianMixture(n_components=GMM_components, covariance_type=cov_type, n_init=1, 
                    tol=1e-6, reg_covar=1e-6, init_params=initialization_method)

gmm.fit(val_embeddings)
joblib.dump(gmm, 'ta_100_gmm.pkl')

train_log_likelihood = gmm.score_samples(val_embeddings)

# Evaluate model on test set
# ==================================================
log_likelihood = gmm.score_samples(test_embeddings)
test_scores = -log_likelihood

roc_auc_gmm = roc_auc_score(test_labels, test_scores)
print(f'ROC AUC score for GMM: {roc_auc_gmm:.4f}')

fpr_gmm, tpr_gmm, thresholds = roc_curve(test_labels, test_scores)

# ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_gmm, tpr_gmm, label=f'GMM (AUC = {roc_auc_gmm:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("ta_100_roc_auc_curve")

optimal_idx = np.argmax(tpr_gmm - fpr_gmm)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold}')

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
plt.savefig("ta_100_cm.png")
plt.show()

# Calculate confusion matrix values
TP = sum((true == 1 and pred == 1) for true, pred in zip(test_labels, artifacts))
TN = sum((true == 0 and pred == 0) for true, pred in zip(test_labels, artifacts))
FP = sum((true == 0 and pred == 1) for true, pred in zip(test_labels, artifacts))
FN = sum((true == 1 and pred == 0) for true, pred in zip(test_labels, artifacts))

print(TP,TN,FP,FN)

# Calculate metrics
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")