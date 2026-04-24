import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical

# ─── Defines the dataset splitter
#
def prepare_dataset(X_train, y_train, X_test, y_test):
    # Convert categorical labels
    le = preprocessing.LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Normalize images
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Add a channel dimension (for grayscale images)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Convert to binary classes
    y_train_onehot = to_categorical(y_train_encoded, num_classes=2)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=2)

    # Define images from each class
    X_train_kidney = X_train[y_train == 1]  # Imagens COM cálculos renais
    y_train_kidney = y_train_onehot[y_train == 1]

    X_train_normal = X_train[y_train == 0]  # Imagens SEM cálculos renais
    y_train_normal = y_train_onehot[y_train == 0]

    X_test_kidney = X_test[y_test == 1]  # Imagens COM cálculos renais
    y_test_kidney = y_test_onehot[y_test == 1]

    X_test_normal = X_test[y_test == 0]  # Imagens SEM cálculos renais
    y_test_normal = y_test_onehot[y_test == 0]

    return X_train_kidney, y_train_kidney, X_train_normal, y_train_normal, X_test_kidney, y_test_kidney, X_test_normal, y_test_normal
