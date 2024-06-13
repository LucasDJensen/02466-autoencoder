import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

import shutil, sys
from datetime import datetime
import matplotlib.pyplot as plt

from Autoencoder import AutoencoderBasic, LSTMAutoencoder, BiLSTMAutoencoder
from config import Config

from load_data import load_data
from _globals import HPC_STORAGE_PATH, HPC_STORAGE_KORNUM_FILE_LIST_PATH

# Parameters
# ==================================================

# Misc Parameters
# My Parameters
# eeg_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg1/test_list.txt") # "file containing the list of test EEG data")
# eog_test_data=os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "eeg2/test_list.txt") # "file containing the list of test EOG data")
# emg_test_data =os.path.join(HPC_STORAGE_KORNUM_FILE_LIST_PATH, "emg/test_list.txt") # "file containing the list of test EMG data")
in_dir="C:/02466/02466-autoencoder/data/kornum_data/"
eeg_test_data=os.path.join(in_dir, "eeg1/test_list.txt") # "file containing the list of test EEG data")
eog_test_data=os.path.join(in_dir, "eeg2/test_list.txt") # "file containing the list of test EOG data")
emg_test_data =os.path.join(in_dir, "emg/test_list.txt") # "file containing the list of test EMG data")


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
        config.artifacts_label = config.nclasses_data - 1 

test_data = load_data( eeg_filelist=os.path.abspath(eeg_test_data),
                        eog_filelist=os.path.abspath(eog_test_data),
                        emg_filelist=os.path.abspath(emg_test_data),
                        data_shape_2=[config.frame_seq_len, config.ndim],
                        seq_len=config.sub_seq_len* config.nsubseq,
                        nclasses = config.nclasses_data, 
                        artifact_detection = config.artifact_detection,
                        artifacts_label = config.artifacts_label)

xtest=test_data.X2
#xtest=tf.reshape(xtest, [-1, config.frame_seq_len, config.ndim,1])

### Open saved model

autoencoder = tf.keras.models.load_model('autoencoder_model.h5')

reconstructions = autoencoder.predict(xtest)

r2_scores=[]
for c in range(3):

    r2_score_channel=[]
    for i in range(xtest.shape[0]):
        r2_score_channel.append(r2_score(tf.squeeze(xtest[i,:,:,c]), tf.squeeze(reconstructions[i,:,:,c])))

    r2_scores.append(r2_score_channel)

print(r2_scores)
average_r2_score = np.mean(r2_scores,axis=1)

print(f'Average R^2 score for each channel: {average_r2_score}')


# loss = tf.keras.losses.mae(reconstructions,xtest)

# plt.hist(np.asarray(loss).flatten(), bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
# plt.show()
#plt.savefig("Test loss distribution")
