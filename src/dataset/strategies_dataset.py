import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator   

# ─── Add random noise to an image for data augmentation
#
def add_noise(img):
    noise = np.random.uniform(-0.02 * 255, 0.02 * 255, img.shape)
    img_noisy = img + noise
    return np.clip(img_noisy, 0, 255)

# ─── Balance classes by generating augmented images from the minority class
#
def apply_augmentation(X_train, y_train, minority_class=1, majority_class=0):
    X_minority = X_train[y_train == minority_class]
    X_majority = X_train[y_train == majority_class]
    
    diff = len(X_majority) - len(X_minority)
    
    if diff <= 0:
        return X_train.copy(), y_train.copy(), np.array([])
        
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=add_noise,
        fill_mode='nearest'
    )
    
    needs_squeeze = False
    if len(X_minority.shape) == 3:
        X_minority = np.expand_dims(X_minority, axis=-1)
        needs_squeeze = True
        
    augmented_X = []
    augmented_y = []
    
    batch_size = min(32, diff)
    generator = datagen.flow(X_minority, np.ones(len(X_minority)) * minority_class, batch_size=batch_size)
    
    while len(augmented_X) < diff:
        batch_X, batch_y = next(generator)
        for i in range(len(batch_X)):
            if len(augmented_X) < diff:
                augmented_X.append(batch_X[i])
                augmented_y.append(batch_y[i])
            else:
                break
                
    augmented_X = np.array(augmented_X)
    augmented_y = np.array(augmented_y)
    
    if needs_squeeze:
        augmented_X = np.squeeze(augmented_X, axis=-1)
        
    X_train_balanced = np.concatenate([X_train, augmented_X], axis=0)
    y_train_balanced = np.concatenate([y_train, augmented_y], axis=0)
    
    # Shuffle
    indices = np.arange(len(X_train_balanced))
    np.random.shuffle(indices)
    
    return X_train_balanced[indices], y_train_balanced[indices], augmented_X

# ─── Balance classes by removing random samples from the majority class
#
def apply_undersampling(X_train, y_train, minority_class=1, majority_class=0):
    X_minority = X_train[y_train == minority_class]
    y_minority = y_train[y_train == minority_class]
    
    X_majority = X_train[y_train == majority_class]
    y_majority = y_train[y_train == majority_class]
    
    if len(X_majority) > len(X_minority):
        kept_indices = np.random.choice(len(X_majority), len(X_minority), replace=False)
        all_indices = np.arange(len(X_majority))
        excluded_indices = np.setdiff1d(all_indices, kept_indices)
        
        X_majority_sampled = X_majority[kept_indices]
        y_majority_sampled = y_majority[kept_indices]
    else:
        X_majority_sampled = X_majority
        y_majority_sampled = y_majority
        excluded_indices = np.array([], dtype=int)
        
    X_train_balanced = np.concatenate([X_minority, X_majority_sampled], axis=0)
    y_train_balanced = np.concatenate([y_minority, y_majority_sampled], axis=0)
    
    # Shuffle
    indices = np.arange(len(X_train_balanced))
    np.random.shuffle(indices)
    
    return X_train_balanced[indices], y_train_balanced[indices], excluded_indices
