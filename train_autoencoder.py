import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import shutil, sys
from datetime import datetime

from Autoencoder import AutoencoderBasic
from config import Config

from load_data import load_data

from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH


# Parameters
# ==================================================
in_dir="C:/02466/02466-autoencoder/data/kornum_data/"

# Data
#eeg_train_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/train_list.txt") #file containing the list of training EEG data
#eeg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
#eog_train_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/train_list.txt") #file containing the list of training EOG data")
#eog_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
#emg_train_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/train_list.txt") #"file containing the list of training EMG data")
#emg_eval_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")

eeg_train_data=os.path.join(in_dir, "eeg1/train_list.txt") #file containing the list of training EEG data
eeg_eval_data=os.path.join(in_dir, "eeg1/eval_list.txt") #file containing the list of evaluation EEG data")
eog_train_data=os.path.join(in_dir, "eeg2/train_list.txt") #file containing the list of training EOG data")
eog_eval_data=os.path.join(in_dir, "eeg2/eval_list.txt") #file containing the list of evaluation EOG data")
emg_train_data=os.path.join(in_dir, "emg/train_list.txt") #"file containing the list of training EMG data")
emg_eval_data=os.path.join(in_dir, "emg/eval_list.txt") #"file containing the list of evaluation EMG data")


config = Config()

if config.artifact_detection == True:
    config.artifacts_label = config.nclasses_data - 1 # right now the code probably just works when the artifact label is the last one
    config.nclasses_model = 1
    config.nclasses_data = 2
    assert config.mask_artifacts == False, "mask_artifacts must be False if artifact_detection=True"
    print('Artifact detection is active. nclasses_data set to 2, nclasses_model set to 1.')
elif config.artifact_detection == False:
    if config.mask_artifacts == True:
        config.artifacts_label = config.nclasses_data - 1 # right now the code probably just works when the artifact label is the last one
        config.nclasses_model = config.nclasses_data - 1 
    else:
        config.nclasses_model = config.nclasses_data
        config.artifacts_label = config.nclasses_data - 1 # right now the code probably just works when the artifact label is the last one


# config.learning_rate = 1e-4 / FLAGS.batch_size # scaling by btach size because now I'm normalizing the loss by the number of elements in batch
#config.l2_reg_lambda = config.l2_reg_lambda / config.batch_size # scaling by btach size because now I'm normalizing the loss by the number of elements in batch

train_data = load_data( eeg_filelist=os.path.abspath(eeg_train_data),
                        eog_filelist=os.path.abspath(eog_train_data),
                        emg_filelist=os.path.abspath(emg_train_data),
                        data_shape_2=[config.frame_seq_len, config.ndim],
                        seq_len=config.sub_seq_len* config.nsubseq,
                        nclasses = config.nclasses_data, 
                        artifact_detection = config.artifact_detection,
                        artifacts_label = config.artifacts_label)

val_data = load_data(eeg_filelist=os.path.abspath(eeg_eval_data),
                    eog_filelist=os.path.abspath(eog_eval_data),
                    emg_filelist=os.path.abspath(emg_eval_data),
                    data_shape_2=[config.frame_seq_len, config.ndim],
                    seq_len=config.sub_seq_len* config.nsubseq,
                    nclasses = config.nclasses_data, 
                    artifact_detection = config.artifact_detection,
                    artifacts_label = config.artifacts_label)


# Training
# ==================================================
xtrain=train_data.X2
xtrain=tf.reshape(xtrain, [-1, config.frame_seq_len, config.ndim,1])
xval=val_data.X2
xval=tf.reshape(xval, [-1, config.frame_seq_len, config.ndim,1])
ABmodel=AutoencoderBasic(config=config,xtrain=xtrain,xval=xval)
autoencoder,encoder=ABmodel.model()
fitted_model=autoencoder.fit(xtrain, xtrain, epochs=config.training_epoch, batch_size=config.batch_size,
                        validation_data=(xval,xval))

autoencoder.save('autoencoder_model.h5')

# Plot training and validation loss:
train_loss = fitted_model.history['loss']
val_loss = fitted_model.history['val_loss']

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.savefig("BasicAutoencoder_train_val_loss.png")

## Obtaining embeddings:
embed = encoder.predict(xtrain)
print("Shape of embedding vectors:", embed.shape)

tsne = TSNE(n_components=2, random_state=42)
reduced_embed = tsne.fit_transform(embed)

# Plot the embeddings
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embed[:, 0], reduced_embed[:, 1], s=5)
plt.title('t-SNE visualization of embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig("tsne_embedding.png")
