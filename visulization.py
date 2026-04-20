import matplotlib.pyplot as plt
import numpy as np

def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    epochs = len(model_history.history['accuracy'])
    step = max(1, epochs // 10)

    # Accuracy
    axs[0].plot(range(1, epochs + 1), model_history.history['accuracy'])
    axs[0].plot(range(1, epochs + 1), model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, epochs + 1, step))
    axs[0].legend(['train', 'val'], loc='best')

    # Loss
    axs[1].plot(range(1, epochs + 1), model_history.history['loss'])
    axs[1].plot(range(1, epochs + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, epochs + 1, step))
    axs[1].legend(['train', 'val'], loc='best')

    plt.show()