import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from filterbank_shape import FilterbankShape

class AutoencoderBasic():
# Basic model does not use LSTM or any advanced techniques
# It just a simple autoencoder with dense layers

    def __init__(self,config):
        self.config = config

        # Input dimension [8,10,17,129,1]
        self.input_x = tf.placeholder(tf.float32, [None,
                                                   self.config.nsubseq,
                                                   self.config.sub_seq_len,
                                                   self.config.frame_seq_len,
                                                   self.config.ndim,
                                                   self.config.nchannel],
                                                   name="input_x")

        self.dropout_rnn = tf.placeholder(tf.float32, name="dropout_rnn")
        self.istraining = tf.placeholder(tf.bool, name='istraining')

        self.filtershape = FilterbankShape()

        self.latent_dim=self.config.latent_dim

        x = tf.reshape(self.input_x, [-1, self.config.ndim, self.config.nchannel])
        processed_x = self.preprocessing(x)
        processed_x = tf.reshape(processed_x, [-1, self.config.frame_seq_len, self.config.nfilter*self.config.nchannel])
        # This gives us the log-magnitude time-frequency images S (T*F) of each epoch

        # Defining autoencoder
        encoder_input=tf.keras.Input(shape=(64,64,3),name="img_in")
        x=tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
        x=tf.keras.layers.MaxPooling2D((2,2),padding="same")(x)
        x=tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
        x=tf.keras.layers.MaxPooling2D((2,2),padding="same")(x)
        x=tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same")
        x=tf.keras.layers.MaxPooling2D((2,2),padding="same")(x)
        x=tf.keras.layers.Flatten()(x)
        encoder_output=tf.keras.layers.Dense(512,activation="relu")(x)

        encoder=tf.keras.Model(encoder_input,encoder_output,name="encoder")

        decoder_input=tf.keras.layers.Dense(2048,activation="relu")(encoder_output)
        x=tf.keras.layers.Reshape((8,8,32))(decoder_input)
        x=tf.keras.layers.Conv2D(32,(3,3),activation="relu",padding="same")
        x=tf.keras.layers.UpSampling2D((2,2),padding="same")(x)
        x=tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
        x=tf.keras.layers.UpSampling2D((2,2),padding="same")(x)
        x=tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
        x=tf.keras.layers.UpSampling2D((2,2),padding="same")(x)
        decoder_output=tf.keras.layers.Conv2D(3,(3,3),activation="relu",padding="same")

        self.encoder=tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim,activation="relu"),
        ])

    def preprocessing(self, input):
            
            # input of shape [-1, ndim, nchannel]
            # triangular filterbank shape
            Wbl = tf.constant(self.filtershape.lin_tri_filter_shape(nfilt=self.config.nfilter,
                                                                    nfft=self.config.nfft,
                                                                    samplerate=self.config.samplerate,
                                                                    lowfreq=self.config.lowfreq,
                                                                    highfreq=self.config.highfreq),
                                                                    dtype=tf.float32,
                                                                    name="W-filter-shape-eeg")

            # filter bank layer for eeg
            with tf.variable_scope("seq_filterbank-layer-eeg", reuse=tf.AUTO_REUSE):
                # Temporarily crush the feature_mat's dimensions
                Xeeg = tf.reshape(tf.squeeze(input[:, :, 0]), [-1, self.config.ndim])
                # first filter bank layer
                Weeg = tf.get_variable('Weeg', shape=[self.config.ndim, self.config.nfilter],
                                       initializer=tf.random_normal_initializer())
                # non-negative constraints
                Weeg = tf.sigmoid(Weeg)
                Wfb_eeg = tf.multiply(Weeg, Wbl)
                HWeeg = tf.matmul(Xeeg, Wfb_eeg)  # filtering

            # filter bank layer for eog
            if (self.config.nchannel > 1):
                with tf.variable_scope("seq_filterbank-layer-eog", reuse=tf.AUTO_REUSE):
                    # Temporarily crush the feature_mat's dimensions
                    Xeog = tf.reshape(tf.squeeze(input[:, :, 1]), [-1, self.config.ndim])
                    # first filter bank layer
                    Weog = tf.get_variable('Weog', shape=[self.config.ndim, self.config.nfilter],
                                           initializer=tf.random_normal_initializer())
                    # non-negative constraints
                    Weog = tf.sigmoid(Weog)
                    Wfb_eog = tf.multiply(Weog, Wbl)
                    HWeog = tf.matmul(Xeog, Wfb_eog)  # filtering

            # filter bank layer for emg
            if (self.config.nchannel > 2):
                with tf.variable_scope("seq_filterbank-layer-emg", reuse=tf.AUTO_REUSE):
                    # Temporarily crush the feature_mat's dimensions
                    Xemg = tf.reshape(tf.squeeze(input[:, :, 2]), [-1, self.config.ndim])
                    # first filter bank layer
                    Wemg = tf.get_variable('Wemg', shape=[self.config.ndim, self.config.nfilter],
                                           initializer=tf.random_normal_initializer())
                    # non-negative constraints
                    Wemg = tf.sigmoid(Wemg)
                    Wfb_emg = tf.multiply(Wemg, Wbl)
                    HWemg = tf.matmul(Xemg, Wfb_emg)  # filtering

            if (self.config.nchannel > 2):
                X2 = tf.concat([HWeeg, HWeog, HWemg], axis=1)
            elif (self.config.nchannel > 1):
                X2 = tf.concat([HWeeg, HWeog], axis=1)
            else:
                X2 = HWeeg

            return X2
    