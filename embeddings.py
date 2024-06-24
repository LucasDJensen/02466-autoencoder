import os
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from load_data import load_data
from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH


# Filelists
# ==================================================
# eeg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
# eog_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
# emg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")

eeg_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt") # "file containing the list of test EEG data")
eog_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt") # "file containing the list of test EOG data")
emg_test_data =os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt") # "file containing the list of test EMG data")

# in_dir="C:/02466/02466-autoencoder/data/kornum_data/"

# eeg_eval_data=os.path.join(in_dir, "eeg1/eval_list.txt")
# eog_eval_data=os.path.join(in_dir, "eeg2/eval_list.txt")
# emg_eval_data=os.path.join(in_dir, "emg/eval_list.txt")

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

# val_data = load_data(eeg_filelist=os.path.abspath(eeg_eval_data),
#                     eog_filelist=os.path.abspath(eog_eval_data),
#                     emg_filelist=os.path.abspath(emg_eval_data),
#                     data_shape_2=[frame_seq_len, ndim],
#                     seq_len=sub_seq_len*nsubseq,
#                     nclasses = nclasses_data, 
#                     artifact_detection = artifact_detection,
#                     artifacts_label = artifacts_label)

# test_data = load_data(eeg_filelist=os.path.abspath(eeg_test_data),
#                     eog_filelist=os.path.abspath(eog_test_data),
#                     emg_filelist=os.path.abspath(emg_test_data),
#                     data_shape_2=[frame_seq_len, ndim],
#                     seq_len=sub_seq_len*nsubseq,
#                     nclasses = 2, 
#                     artifact_detection = True,
#                     artifacts_label = (nclasses_data - 1))

test_data = load_data(eeg_filelist=os.path.abspath(eeg_test_data),
                    eog_filelist=os.path.abspath(eog_test_data),
                    emg_filelist=os.path.abspath(emg_test_data),
                    data_shape_2=[frame_seq_len, ndim],
                    seq_len=sub_seq_len*nsubseq,
                    nclasses = nclasses_data, 
                    artifact_detection = artifact_detection,
                    artifacts_label = artifacts_label)

# xval = val_data.X2
xtest = test_data.X2[:-1,:,:,:]
# val_labels = val_data.y[:,:,0]
print(sum(test_labels[:,3]))
test_labels = test_data.y[:-1,:,0] 

print(sum(test_labels[:,3]))
# fxval=tf.reshape(xval,(-1,17*129*3))
fxtest=tf.reshape(xtest,(-1,17*129*3))

#autoencoder = tf.keras.models.load_model('dense_ae.h5')
#encoder=tf.keras.Model(inputs=autoencoder.input,outputs=autoencoder.layers[4].output)
serialized_tensor = tf.io.read_file('dense_100_ae_features.tfrecord')
val_features = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

#test_features = encoder.predict(xtest)
mean_features = tf.reduce_mean(val_features,axis=0)
std_features = tf.math.reduce_std(val_features,axis=0)

# val_embeddings = (val_features - mean_features)/std_features
test_embeddings = (test_features - mean_features)/std_features

# t-SNE
# ==================================================

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(val_embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[test_labels[:,0,0] == 1, 0], embeddings_2d[test_labels[:,0,0] == 1, 1],color="r",label="NREM")
plt.scatter(embeddings_2d[test_labels[:,1,0] == 1, 0], embeddings_2d[test_labels[:,1,0] == 1, 1],color="b",label="REM")
plt.scatter(embeddings_2d[test_labels[:,2,0] == 1, 0], embeddings_2d[test_labels[:,2,0] == 1, 1],color="g",label="WAKE")
plt.scatter(embeddings_2d[test_labels[:,3,0] == 1, 0], embeddings_2d[test_labels[:,3,0] == 1, 1],color="pink",label="Artifact")
plt.legend()
plt.title('t-SNE Visualization of Embedding Vectors')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig("tsne_dense600.png")
plt.show()
