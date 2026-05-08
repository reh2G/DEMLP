import os
import glob
import cv2
import numpy as np

from src.dataset.utils_dataset import sort_files_numerically
from src.dataset.groups_dataset import find_groups, save_groups

# ─── Defines the dataset splitter
#
def read_dataset(path, name, DEBUG, SIMILARITY, output_dir):
    img_types = ['*.jpg', '*.png']
    classes = {'healthy': 0, 'stone': 1}

    all_images, all_labels, all_paths, all_groups = [], [], [], []
    group_offset = 0  # unique IDs for different classes

    for class_name, label in classes.items():
        class_path = os.path.join(path, class_name)
        images, labels, paths = [], [], []

        print(f'\nLendo imagens {class_name} de: {class_path}')

        for pattern in img_types:
            img_paths = glob.glob(os.path.join(class_path, pattern))
            img_paths = sort_files_numerically(file_paths=img_paths)

            for img_path in img_paths:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(label)
                paths.append(img_path)

        images = np.array(images)
        labels = np.array(labels)
        paths = np.array(paths)

        print(f'Total {class_name}: {len(images)} imagens')

        group_ids = find_groups(images=images, image_paths=paths, similarity_threshold=SIMILARITY, DEBUG=DEBUG)

        group_ids += group_offset
        group_offset = group_ids.max() + 1

        print(f'Grupos encontrados em {class_name}: {len(np.unique(group_ids))}')

        all_images.extend(images)
        all_labels.extend(labels)
        all_paths.extend(paths)
        all_groups.extend(group_ids)

    X = np.array(all_images)
    y = np.array(all_labels)
    paths = np.array(all_paths)
    groups = np.array(all_groups)

    save_groups(image_paths=paths, group_ids=groups, output_dir=output_dir)

    print(f'\nTotal: {len(X)} imagens | {len(np.unique(groups))} grupos únicos')

    if DEBUG:
        with np.printoptions(threshold=np.inf):
            print(groups)

    return X, y, groups
