# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:58:35 2024

@author: bella
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

plt.imshow(x_train[1])

#standize data
x_train=x_train/255
x_test=x_test/255

#encoder
encoder_input=keras.Input(shape=(28,28,1),name="img")
#condese data
x=keras.layers.Flatten()(encoder_input)
encoder_output= keras.layers.Dense(64,activation="relu")(x)

encoder=keras.Model(encoder_input,encoder_output,name="encoder")

decoder_input=keras.layers.Dense(784,activation="relu")(encoder_output)
#x=keras.layers.Dense(784,activation="relu")(decoder_input)
decoder_output=keras.layers.Reshape((28,28,1))(decoder_input)

opt=keras.optimizers.Adam(learning_rate=0.001)
autoencoder=keras.Model(encoder_input,decoder_output,name="autoencoder")

autoencoder.summary()
#compile model
autoencoder.compile(opt,loss="mse")
autoencoder.fit(x_train,x_train,epochs=3,batch_size=32,validation_split=0.1)

example=encoder.predict([x_test[0].reshape(-1,28,28,1)])[0]
print(example)
#not ment to be an image but:
plt.figure()
plt.subplot(1,2,1)
plt.imshow(x_test[0])
plt.subplot(1,2,2)


ae_out=autoencoder.predict([x_test[0].reshape(-1,28,28,1)])[0]
plt.imshow(ae_out)

# Add noise
def add_noise(img,random_chance=5):
    noisy=[]
    for row in img:
        new_row=[]
        for pix in row:
            if random.choice(range(100))<=random_chance:
                new_val=random.uniform(0,1)
                new_row.append(new_val)
            else:
                new_row.append(pix)
        noisy.append(new_row)
    return np.array(noisy)

plt.figure()
noisy=add_noise(x_test[0])
plt.subplot(1,2,1)
plt.imshow(noisy)
plt.subplot(1,2,2)
ae_out=autoencoder.predict([x_test[0].reshape(-1,28,28,1)])[0]
plt.imshow(ae_out)


"""Convolutional Autoencoder"""
encoder_input=keras.Input(shape=(64,64,3),name="img_in")
x=keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
x=keras.layers.MaxPooling2D((2,2),padding="same")(x)
x=keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
x=keras.layers.MaxPooling2D((2,2),padding="same")(x)
x=keras.layers.Conv2D(32,(3,3),activation="relu",padding="same")
x=keras.layers.MaxPooling2D((2,2),padding="same")(x)
x=keras.layers.Flatten()(x)
encoder_output=keras.layers.Dense(512,activation="relu")(x)

encoder=keras.Model(encoder_input,encoder_output,name="encoder")

decoder_input=keras.layers.Dense(2048,activation="relu")(encoder_output)
x=keras.layers.Reshape((8,8,32))(decoder_input)
x=keras.layers.Conv2D(32,(3,3),activation="relu",padding="same")
x=keras.layers.UpSampling2D((2,2),padding="same")(x)
x=keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
x=keras.layers.UpSampling2D((2,2),padding="same")(x)
x=keras.layers.Conv2D(64,(3,3),activation="relu",padding="same")
x=keras.layers.UpSampling2D((2,2),padding="same")(x)
decoder_output=keras.layers.Conv2D(3,(3,3),activation="relu",padding="same")

## Inception block (with torch)
class DeepSeparator(nn.Module):

    def __init__(self):
        super(DeepSeparator, self).__init__()

        self.conv1_1_1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv1_1_2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=5, padding=2)
        self.conv1_1_3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=11, padding=5)
        self.conv1_1_4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=15, padding=7)

def forward(self, x, indicator):

        emb_x = x
        learnable_atte_x = x

        emb_x = torch.unsqueeze(emb_x, 1)

        emb_x_1 = self.conv1_1_1(emb_x)
        emb_x_2 = self.conv1_1_2(emb_x)
        emb_x_3 = self.conv1_1_3(emb_x)
        emb_x_4 = self.conv1_1_4(emb_x)

        emb_x = torch.cat((emb_x_1, emb_x_2, emb_x_3, emb_x_4), dim=1)
        emb_x = torch.relu(emb_x)