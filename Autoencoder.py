import tensorflow as tf

class Autoencoder():
# Simple autoencoder based on convolutional layers

    def __init__(self, input_shape = (17,129,3), activation_function = "elu", latent_dim = 1000, learning_rate = 0.0001):

        self.input_shape = input_shape
        self.activ_function = activation_function
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
        
    def autoencoder(self):

        # Encoder
        encoder_input = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (2,2), activation=self.activ_function)(encoder_input)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(64, (3,3), activation=self.activ_function)(x)
        x = tf.keras.layers.MaxPooling2D((2,2))(x)
        x = tf.keras.layers.Conv2D(128, (3,3), activation=self.activ_function)(x)
        shape_before_flattening = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(self.latent_dim, activation=self.activ_function)(x)

        # Decoder
        x = tf.keras.layers.Dense(units=tf.reduce_prod(shape_before_flattening[1:]), activation=self.activ_function)(encoder_output)
        x = tf.keras.layers.Reshape(target_shape=shape_before_flattening[1:])(x)
        x = tf.keras.layers.Conv2DTranspose(128, (3,3), activation=self.activ_function)(x)
        x = tf.keras.layers.Conv2DTranspose(64, (2, 2), activation=self.activ_function)(x)
        x = tf.keras.layers.UpSampling2D((2,2))(x)
        x = tf.keras.layers.Conv2DTranspose(32, (2, 2), activation=self.activ_function)(x)
        x = tf.keras.layers.UpSampling2D((2,2))(x)
        decoded = tf.keras.layers.Conv2D(3, (2, 2), activation='linear')(x)

        # Compile
        model = tf.keras.Model(encoder_input, decoded, name="autoencoder")
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(opt, loss="mse")

        self.summary=model.summary()
        
        return model
    
class TemporalAutoencoder():

    def __init__(self, input_shape = (512,3), activation_function = "elu", latent_dim = 1000, learning_rate = 0.0001):

        self.input_shape = input_shape
        self.activ_function = activation_function
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
    
    def autoencoder(self):
        
        # Encoder
        input = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv1D(32, 3, activation=self.activ_function)(input) #N
        x = tf.keras.layers.MaxPooling1D(2)(x) #N
        x = tf.keras.layers.Conv1D(64, 6, activation=self.activ_function)(x)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        x = tf.keras.layers.Conv1D(128, 8, activation=self.activ_function)(x)
        x = tf.keras.layers.Conv1D(256, 8, activation=self.activ_function, strides=2)(x)
        shape_before_flattening = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(self.latent_dim, activation=self.activ_function)(x)

        # Decoder
        x = tf.keras.layers.Dense(units=tf.reduce_prod(shape_before_flattening[1:]), activation=self.activ_function)(encoder_output)
        x = tf.keras.layers.Reshape(target_shape=shape_before_flattening[1:])(x)
        x = tf.keras.layers.Conv1DTranspose(128, 9, activation=self.activ_function,strides=2)(x)
        x = tf.keras.layers.Conv1DTranspose(64, 8, activation=self.activ_function)(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1DTranspose(32, 5, activation=self.activ_function)(x)
        x = tf.keras.layers.UpSampling1D(2)(x)
        decoder_output = tf.keras.layers.Conv1D(3, 3, activation='linear',padding="same")(x)

        # Compile
        model = tf.keras.Model(inputs=input,outputs=decoder_output)
        opt=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(opt,loss="mse")
        self.summary=model.summary()
        
        return model

class DenseAutoencoder():

    def __init__(self, input_shape = (512*3), activation_function = "elu", latent_dim = 1000, learning_rate = 0.0001):
        
        self.input_shape = input_shape
        self.activ_function = activation_function
        self.learning_rate = learning_rate
        self.latent_dim = latent_dim
    
    def autoencoder(self):

        # Encoder
        inputs = tf.keras.Input(shape = self.input_shape)
        x = tf.keras.layers.Dense(2048, activation=self.activ_function)(inputs)
        x = tf.keras.layers.Dense(1024, activation=self.activ_function)(x)
        x = tf.keras.layers.Dense(768, activation=self.activ_function)(x)
        encoder_output = tf.keras.layers.Dense(self.latent_dim, activation=self.activ_function)(x)

        # Decoder
        x = tf.keras.layers.Dense(768, activation=self.activ_function)(encoder_output)
        x = tf.keras.layers.Dense(1024, activation=self.activ_function)(x)
        x = tf.keras.layers.Dense(2048, activation=self.activ_function)(x)
        decoder_output = tf.keras.layers.Dense(self.input_shape[0], activation='linear')(x)

        # Autoencoder
        model = tf.keras.Model(inputs=inputs,outputs=decoder_output)
        opt=tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(opt,loss="mse")
        self.summary=model.summary()
        
        return model
