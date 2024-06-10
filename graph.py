import tensorflow as tf
import matplotlib.pyplot as plt

# Load trained model
autoencoder = tf.keras.models.load_model('BiLSTMautoencoder_model.h5')

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

