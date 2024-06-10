import numpy as np
import tensorflow as tf


class AutoencoderBasic():
# Simple autoencoder based on convolutional layers

    def __init__(self, config, xtrain, xval):
        self.config = config

        height, width = np.shape(xtrain)[1:3]
        self.input_shape = (height, width, 1)
        #self.autoencoder=self.model()
        
    def model(self):

        # Encoder
        encoder_input = tf.keras.Input(shape=self.input_shape, name="img_in") # (17,129,1)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (0, 0)))(encoder_input) # (18,129,1)
        
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # Output: (18, 129, 32)
        x = tf.keras.layers.MaxPooling2D((3, 3), padding='same')(x)  # Output: (6, 43, 32)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # Output: (6, 43, 64)
        xs = np.shape(x)
        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(self.config.latent_dim, activation="relu")(x)
        encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

        # Decoder
        x = tf.keras.layers.Dense(units=xs[1]*xs[2]*xs[3], activation="relu")(encoder_output)
        x = tf.keras.layers.Reshape((xs[1], xs[2], xs[3]))(x)
        x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)  # Output: (9, 66, 64)
        x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = tf.keras.layers.UpSampling2D((3, 3))(x)
        x = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Output: (17, 129, 1)
        decoded = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)))(x)  # Output: (17, 129, 1)
    
        autoencoder = tf.keras.Model(encoder_input, decoded, name="autoencoder")
        opt = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        autoencoder.compile(opt, loss="mse") # for now we use MSE

        print(autoencoder.summary())
        return autoencoder,encoder

class LSTMAutoencoder():

    def __init__(self, config, xtrain, xval):

        self.config = config
        self.input_shape = np.shape(xtrain)[1:3]
        self.autoencoder = self.model()
    
    def model(self):
        input = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.LSTM(64, activation="relu", return_sequences=True)(input)
        x = tf.keras.layers.LSTM(32, activation="relu")(x)

        x = tf.keras.layers.RepeatVector(self.input_shape[0])(x)

        x = tf.keras.layers.LSTM(32, activation="relu", return_sequences=True)(x)
        x = tf.keras.layers.LSTM(64, activation="relu", return_sequences=True)(x)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_shape[1]))(x)

        model = tf.keras.Model(inputs=input,outputs=output)
        opt=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(opt,loss="mse")

        print(model.summary())
        return model

class BiLSTMAutoencoder(LSTMAutoencoder):

    def __init__(self, config, xtrain, xval):
        super().__init__(config, xtrain, xval)
    
    def model(self):
        input = tf.keras.Input(shape=(self.input_shape))
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation="relu"))(input)

        x = tf.keras.layers.RepeatVector(self.input_shape[0])(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation="relu",return_sequences=True))(x)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.input_shape[1]))(x)

        model = tf.keras.Model(inputs=input,outputs=output)
        opt = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(opt,loss="mse")

        print(model.summary())
        return model
