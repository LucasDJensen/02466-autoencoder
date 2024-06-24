import matplotlib.pyplot as plt
import numpy as np

import re

file = r'C:\02466\02466-autoencoder\Final results\training_ta_100_22071589.out'
training_losses = []
validation_losses = []
with open(file, 'r') as f:
    for line in f.readlines():
        matches = re.findall(r"loss: (\d+\.\d+) - val_loss: (\d+\.\d+)", line)
        if matches:
            loss, val_loss = matches[0]
            training_losses.append(float(loss))
            validation_losses.append(float(val_loss))

epochs = np.arange(1, len(training_losses) + 1)

# Plotting the training and test loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_losses, label='Training Loss')
plt.plot(epochs, validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig("train_val_loss_curve_ta100.png")