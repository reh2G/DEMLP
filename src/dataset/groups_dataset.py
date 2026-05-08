import os
import numpy as np
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

# ─── Calculate the correlation coefficient and separe in groups
#
def find_groups(images, image_paths, similarity_threshold, DEBUG=False):
    groups = []
    current_group = [0]

    def are_similar(img1, img2, threshold):
        similarity = ssim(img1, img2)
        return similarity >= threshold

    for i in range(1, len(images)):
        if are_similar(images[i - 1], images[i], similarity_threshold):
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

    if current_group:
        groups.append(current_group)

    if DEBUG:
        for gid, group in enumerate(groups):
            print(f"Grupo {gid + 1}: {len(group)} imagens")
            for idx in group:
                print(f"  {image_paths[idx]}")

    group_ids = np.zeros(len(image_paths), dtype=int)
    for group_id, group_indices in enumerate(groups):
        for idx in group_indices:
            group_ids[idx] = group_id

    return group_ids  # shape: (n_images,)

# ─── Function to save the groups in a text file
#
def save_groups(image_paths, group_ids, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'detected_groups.txt')

    groups_dict = defaultdict(list)
    for path, gid in zip(image_paths, group_ids):
        groups_dict[gid].append(path)

    with open(output_path, 'w') as f:
        for gid, paths in sorted(groups_dict.items()):
            f.write(f'- group_{gid}:\n')
            for p in paths:
                f.write(f'  {p}\n')

    print(f'\nGrupos salvos em {output_path}')
