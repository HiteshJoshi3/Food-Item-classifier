import numpy as np

def probas_to_classes(y_pred):
    """
    Convert model probability outputs to class indices
    """
    return np.argmax(y_pred, axis=1)