import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from config import Config
from load_data import load_data
from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH

in_dir="C:/02466/02466-autoencoder/data/kornum_data/"

# Data
#eeg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
#eog_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
#emg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")

eeg_eval_data=os.path.join(in_dir, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
eog_eval_data=os.path.join(in_dir, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
emg_eval_data=os.path.join(in_dir, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")

# Load trained model
#autoencoder = tf.keras.models.load_model('1d_autoencoder_model.h5')

#print(autoencoder.summary())
# autoencoder=tf.keras.models.load_model('temporal_autoencoder.h5')
# print(autoencoder.summary())

"""
# Plot training and validation loss:
train_loss = autoencoder.history['loss']
val_loss = autoencoder.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

"""
config = Config()

val_data = load_data(eeg_filelist=os.path.abspath(eeg_eval_data),
                    eog_filelist=os.path.abspath(eog_eval_data),
                    emg_filelist=os.path.abspath(emg_eval_data),
                    data_shape_2=[config.frame_seq_len, config.ndim],
                    seq_len=config.sub_seq_len* config.nsubseq,
                    nclasses = config.nclasses_data, 
                    artifact_detection = False,
                    artifacts_label = (config.nclasses_data - 1))

val_labels = val_data.label[:,0]

serialized_tensor=tf.io.read_file('features.tfrecord')
features = tf.io.parse_tensor(serialized_tensor, out_type=tf.float32)

mean_features = tf.reduce_mean(features,axis=0)
std_features = tf.math.reduce_std(features,axis=0)

embeddings = (features - mean_features)/std_features

c=[]
for label in val_labels:
    if label==0:
        c.append('r')
    elif label==1:
        c.append('b')
    elif label==2:
        c.append('g')

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5,c=val_labels)
plt.colorbar()
plt.title('t-SNE Visualization of Embedding Vectors')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.savefig("tsne_labels_embeddings.png")
plt.show()
