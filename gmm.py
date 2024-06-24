import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from config import Config
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
import numpy as np
import seaborn as sns
import joblib

from load_data import load_data

from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH

in_dir="C:/02466/02466-autoencoder/data/kornum_data/"

# eeg_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt") # "file containing the list of test EEG data")
# eog_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt") # "file containing the list of test EOG data")
# emg_test_data =os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt") # "file containing the list of test EMG data")

eeg_test_data=os.path.join(in_dir, "eeg1/test_list.txt") # "file containing the list of test EEG data")
eog_test_data=os.path.join(in_dir, "eeg2/test_list.txt") # "file containing the list of test EOG data")
emg_test_data =os.path.join(in_dir, "emg/test_list.txt") # "file containing the list of test EMG data")

config = Config()

test_data = load_data(eeg_filelist=os.path.abspath(eeg_test_data),
                    eog_filelist=os.path.abspath(eog_test_data),
                    emg_filelist=os.path.abspath(emg_test_data),
                    data_shape_2=[config.frame_seq_len, config.ndim],
                    seq_len=config.sub_seq_len* config.nsubseq,
                    nclasses = 2, 
                    artifact_detection = True,
                    artifacts_label = (config.nclasses_data - 1))

xtest = test_data.X1
xtest = tf.reshape(xtest, [-1,512, 3])
test_labels = test_data.y[:,1,0] 

print("test labels")
print(test_labels)

serialized_tensor=tf.io.read_file('600features.tfrecord')
features = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

mean_features = tf.reduce_mean(features,axis=0)
std_features = tf.math.reduce_std(features,axis=0)

embeddings = (features - mean_features)/std_features

# Do PCA on embeddings
#pca = PCA()
#pca.fit(embeddings)
#projected_data = pca.transform(embeddings)

# Access explained variance ratio
#explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
#cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# plt.figure(figsize=(10, 5))
# plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Variance Explained')
# plt.title('Cumulative Explained Variance')
# plt.grid()
# plt.tight_layout()
# plt.show()

# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# gmm_components= [3,4,5,7,9,15,20,50]
# bics = {n_components: [] for n_components in gmm_components}
# aics = {n_components: [] for n_components in gmm_components}

# for train_index, test_index in kf.split(embeddings):
#     X_train, X_test = tf.gather(embeddings, train_index), tf.gather(embeddings, test_index)

#     for n_components in gmm_components:
#         gmm = GaussianMixture(n_components=n_components, covariance_type='diag', n_init=1, 
#                       tol=1e-6, reg_covar=1e-6, init_params="random")
#         gmm.fit(X_train)
#         bic = gmm.bic(X_test)
#         aic = gmm.aic(X_test)
#         bics[n_components].append(bic)
#         aics[n_components].append(aic)

# Calculate mean BIC and AIC for each number of components
# mean_bics = {n_components: tf.reduce_mean(bics[n_components]) for n_components in gmm_components}
# mean_aics = {n_components: tf.reduce_mean(aics[n_components]) for n_components in gmm_components}

# for n_components in gmm_components:
#     print(f"Components: {n_components}, Mean BIC: {mean_bics[n_components]:.3f}, Mean AIC: {mean_aics[n_components]:.3f}")


optimal_number_of_components = 15
gmm = GaussianMixture(n_components=optimal_number_of_components, covariance_type='diag', n_init=1, 
                      tol=1e-6, reg_covar=1e-6, init_params="random")

#kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
#kde.fit(embeddings)
#joblib.dump(kde, 'kde_model.pkl')

gmm.fit(embeddings)
joblib.dump(gmm, 'optimal_gmm_model.pkl')
#gmm = joblib.load('optimal_gmm_model.pkl')

# Compute log-likelihood of the training data
train_log_likelihood = gmm.score_samples(embeddings)
#train_log_likelihood = kde.score_samples(embeddings)

# Test accuracy on using test data
autoencoder=tf.keras.models.load_model('ta.h5')
encoder=tf.keras.Model(inputs=autoencoder.input,outputs=autoencoder.layers[8].output)

test_data = encoder.predict(xtest)
test_data = (test_data - mean_features)/std_features

log_likelihood = gmm.score_samples(test_data)
# log_likelihood = kde.score_samples(test_data)
test_scores = -log_likelihood

roc_auc_gmm = roc_auc_score(test_labels, test_scores)
print(f'ROC AUC score for GMM: {roc_auc_gmm:.4f}')

# Step 5: Compute the ROC curve
fpr_gmm, tpr_gmm, thresholds = roc_curve(test_labels, test_scores)

# Step 6: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_gmm, tpr_gmm, label=f'GMM (AUC = {roc_auc_gmm:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("600_roc_auc_curve")
#plt.savefig("kde_roc_auc_curve")

optimal_idx = np.argmax(tpr_gmm - fpr_gmm)
optimal_threshold = thresholds[optimal_idx]
print(f'Optimal Threshold: {optimal_threshold}')

# Detect artifacts
artifacts = test_scores < optimal_threshold

print("Artifacts")
print(artifacts)
print("Number of artifact samples detected:", len(artifacts))

# Confusion matrix and scores
plt.figure()
cm = confusion_matrix(test_labels,artifacts)
sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=['not art.','art.'],
            yticklabels=['art.','not art.'],
            cmap="Blues")
plt.xlabel('Prediction',fontsize=13)
plt.ylabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
# # plt.savefig("kde_cm")
plt.savefig("600cm")
plt.show()

print(thresholds)

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

# Print results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

