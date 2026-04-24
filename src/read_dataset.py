from src.utils_dataset import sort_files_numerically
from src.groups_dataset import find_groups, save_groups

import os
import glob
import cv2
import numpy as np

# ─── Configurations
#
DEBUG = False
#DEBUG = True

output_folder = 'output/groups'
type = 'txt'

TEST_SIZE = 0.2
SIMILARITY = 0.75

# ─── Defines the dataset splitter
#
def read_dataset(path, name):
    print(f'Reading dataset...\n')
    
    img_type = ['*.jpg', '*.png']
    images = []
    labels = []
    image_paths = []

# Healthy
    
    healthy_path = os.path.join(path, 'healthy')  # Path to Healthy base images
    healthy_images = []
    healthy_labels = []
    healthy_image_paths = []
    print(f'Reading Healthy images from: {healthy_path}')
    
    for img_type_pattern in img_type:
        img_paths = glob.glob(os.path.join(healthy_path, img_type_pattern))
        img_paths = sort_files_numerically(file_paths=img_paths)  # Ensure the images are sorted numerically

        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))
            healthy_images.append(img)
            healthy_labels.append(0)  # Healthy -> 0
            healthy_image_paths.append(img_path)

    healthy_images = np.array(healthy_images)
    healthy_labels = np.array(healthy_labels)
    healthy_image_paths = np.array(healthy_image_paths)

# Healthy Groups

    healthy_total_images = len(healthy_images)
    healthy_test_size = int(healthy_total_images * TEST_SIZE)

    print(f'Total number of healthy images: {healthy_total_images}')
    print(f'Calculated healthy test size: {healthy_test_size}')

    healthy_train_set, healthy_test_set = find_groups(images=healthy_images, image_paths=healthy_image_paths, test_size=healthy_test_size, similarity_threshold=SIMILARITY, DEBUG=DEBUG)

    print(f'Number of healthy images in train set: {len(healthy_train_set)}')
    print(f'Number of healthy images in test set: {len(healthy_test_set)}\n')

# Kidney stone
    
    kidney_stone_path = os.path.join(path, 'stone')  # Path to Kidney_stone base images
    kidney_stone_images = []
    kidney_stone_labels = []
    kidney_stone_image_paths = []
    print(f'Reading Kidney_stone images from: {kidney_stone_path}')
    
    for img_type_pattern in img_type:
        img_paths = glob.glob(os.path.join(kidney_stone_path, img_type_pattern))
        img_paths = sort_files_numerically(file_paths=img_paths)  # Ensure the images are sorted numerically

        for img_path in img_paths:
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (224, 224))
            kidney_stone_images.append(img)
            kidney_stone_labels.append(1)  # Kidney_stone -> 1
            kidney_stone_image_paths.append(img_path)

    kidney_stone_images = np.array(kidney_stone_images)
    kidney_stone_labels = np.array(kidney_stone_labels)
    kidney_stone_image_paths = np.array(kidney_stone_image_paths)

# Kidney Groups
    
    kidney_stone_total_images = len(kidney_stone_images)
    kidney_stone_test_size = int(kidney_stone_total_images * TEST_SIZE)

    print(f'Total number of kidney images: {kidney_stone_total_images}')
    print(f'Calculated kidney test size: {kidney_stone_test_size}')

    kidney_stone_train_set, kidney_stone_test_set = find_groups(images=kidney_stone_images, image_paths=kidney_stone_image_paths, test_size=kidney_stone_test_size, similarity_threshold=SIMILARITY, DEBUG=DEBUG)
    
    train_groups = [healthy_train_set, kidney_stone_train_set]
    test_groups = [healthy_test_set, kidney_stone_test_set]
    
    base_name = 'groups_' + name
    save_groups(train_groups=train_groups, test_groups=test_groups, base_name=base_name)

    print(f'Number of kidney images in train set: {len(kidney_stone_train_set)}')
    print(f'Number of kidney images in test set: {len(kidney_stone_test_set)}')

# ...
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for img_path in healthy_train_set:
        idx = np.where(healthy_image_paths == img_path)[0][0]
        X_train.append(healthy_images[idx])
        y_train.append(healthy_labels[idx])

    for img_path in healthy_test_set:
        idx = np.where(healthy_image_paths == img_path)[0][0]
        X_test.append(healthy_images[idx])
        y_test.append(healthy_labels[idx])

    for img_path in kidney_stone_train_set:
        idx = np.where(kidney_stone_image_paths == img_path)[0][0]
        X_train.append(kidney_stone_images[idx])
        y_train.append(kidney_stone_labels[idx])

    for img_path in kidney_stone_test_set:
        idx = np.where(kidney_stone_image_paths == img_path)[0][0]
        X_test.append(kidney_stone_images[idx])
        y_test.append(kidney_stone_labels[idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test
