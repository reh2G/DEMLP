from src.utils_dataset import get_next_filename

import os
import random
from skimage.metrics import structural_similarity as ssim

# ─── Configurations
#
output_folder = 'output/groups'

# ─── Calculate the correlation coefficient and separe in groups
#
def find_groups(images, image_paths, test_size, similarity_threshold, DEBUG):
    def are_similar(img1, img2, threshold):
        similarity = ssim(img1, img2)
        return similarity >= threshold

    groups = []
    current_group = [image_paths[0]]

    for i in range(1, len(images)):
        if are_similar(images[i-1], images[i], similarity_threshold):
            current_group.append(image_paths[i])
        else:

            if DEBUG:
                print(f"Group {len(groups)+1} ended with {len(current_group)} images:")
                for img_path in current_group:
                    print(f"  {img_path}")

            groups.append(current_group)
            current_group = [image_paths[i]]
    
    if current_group:

        if DEBUG:
            print(f"Final group {len(groups)+1} with {len(current_group)} images")
            for img_path in current_group:
                print(f"  {img_path}")

        groups.append(current_group)

    all_images = [img_path for group in groups for img_path in group]

    for img_path in image_paths:
        if img_path not in all_images:
            groups.append([img_path])

    random.shuffle(groups)

    train_set = []
    test_set = []
    current_test_size = 0
    
    for i, group in enumerate(groups):

        if DEBUG:
            print(f"Processing number {i+1} with {len(group)} images")

        if current_test_size + len(group) <= test_size:
            test_set.extend(group)
            current_test_size += len(group)
        else:
            train_set.extend(group)

    return train_set, test_set

# ─── Function to save the groups in a text file
#
def save_groups(train_groups, test_groups, base_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_filename = get_next_filename(output_folder=output_folder, base_name=base_name, type=type)
    output_path = os.path.join(output_folder, output_filename)

    with open(output_path, 'w') as file:
        file.write('- test:\n')
        for idx, group in enumerate(test_groups, start=1):
            for img in group:
                file.write(f"{img}\n")
        
        file.write('- train:\n')
        for idx, group in enumerate(train_groups, start=1):
            for img in group:
                file.write(f'{img}\n')
    
    print(f'Group information saved in {output_path}')
