import numpy as np
from keras.utils import to_categorical

# ─── Normalize images, add channel dimension and convert labels to one-hot encoding
#
def prepare_dataset(X_train, y_train, X_val, y_val):
    # Normalize images
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # Add a channel dimension (for grayscale images)
    if len(X_train.shape) == 3:
        X_train = np.expand_dims(X_train, axis=-1)
    if len(X_val.shape) == 3:
        X_val = np.expand_dims(X_val, axis=-1)

    # Convert to one-hot encoding for categorical crossentropy
    y_train_onehot = to_categorical(y_train, num_classes=2)
    y_val_onehot = to_categorical(y_val, num_classes=2)

    return X_train, y_train_onehot, X_val, y_val_onehot
