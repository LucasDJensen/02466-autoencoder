import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import model_selection
from sklearn.metrics import r2_score
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
xtrain = train_data.X2
#xtrain = tf.reshape(xtrain, [-1, config.frame_seq_len, config.ndim, 1])

xval = val_data.X2
#xval = tf.reshape(xval, [-1, config.frame_seq_len, config.ndim, 1])

# K-fold crossvalidation
K = 10 #10
CV = model_selection.KFold(n_splits=K, shuffle=True)

latent_dimensions=tf.range(10, 101, 10)

r2_scores=[]
k=0
for train_index, val_index in CV.split(xtrain):
    print("Cross-validation round ",k)
    x_train, x_val = tf.gather(xtrain, train_index), tf.gather(xtrain, val_index)

    r2_scores_K_fold=[]

    for _,ld in enumerate(latent_dimensions):

        ABmodel = AutoencoderBasic(config=config,xtrain=x_train,xval=x_val,latent_dim=ld)
        autoencoder, encoder = ABmodel.model()
        fitted_model = autoencoder.fit(x_train, x_train, epochs=config.training_epoch, batch_size=config.batch_size,
                            validation_data=(x_val, x_val))
        
        x_val_recon=autoencoder.predict(x_val)

        # Create a list of average r2 score for every channel
        fold_r2_score=[]

        for channel in range(3):
            # Calculate r2 score for every pixel
            fold_channel_r2_score=[]
            for i in range(x_val.shape[0]):
                fold_channel_r2_score.append((tf.squeeze(x_val[i, :, :, channel]), tf.squeeze(x_val_recon[i, :, :, channel])))

            # Calculate average r2 score for that channel
            fold_r2_score.append(tf.reduce_mean(fold_channel_r2_score))

        r2_scores_K_fold.append(fold_r2_score)
    
    r2_scores.append(r2_scores_K_fold)

    k+=1


# Average R^2 score across all folds
average_r2_score_cv = tf.reduce_mean(r2_scores,axis=0)

plt.figure()
plt.plot(average_r2_score_cv)
plt.xlabel("Latent dimension")
plt.ylabel("Channel")
plt.title("Average r2 score across latent dimensions and channels")
plt.savefig("r2_scores.png")

average_r2_score_channels=tf.reduce_mean(average_r2_score_cv,axis=1)
best_ld=latent_dimensions[tf.argmin(average_r2_score_channels)]

# Retrain on entire dataset
ABmodel = AutoencoderBasic(config=config,xtrain=xtrain,xval=xval,latent_dim=best_ld)
autoencoder, encoder = ABmodel.model()
fitted_model = autoencoder.fit(xtrain, xtrain, epochs=config.training_epoch, batch_size=config.batch_size,
                        validation_data=(xval, xval))

autoencoder.save('cv_autoencoder_model.h5')


## Obtain embeddings/features:
"""
latent = encoder.predict(xval)
print("Shape of embedding vectors:", latent.shape)
x,y,z=latent.shape[:3]
latent = tf.reshape(latent, (x, -1))

# Calculate mean and standard deviation acrosss features
means = tf.reduce_mean(latent, axis=0)
stds = tf.reduce_std(latent, axis=0)

norm_latent= (latent - means)/stds


tsne = TSNE(n_components=2, random_state=42)
reduced_embed = tsne.fit_transform(latent)

# Plot the embeddings
plt.figure(figsize=(10, 6))
plt.scatter(reduced_embed[:, 0], reduced_embed[:, 1], s=5)
plt.title('t-SNE visualization of embeddings')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.savefig("tsne_embedding.png")
"""