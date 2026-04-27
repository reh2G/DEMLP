from src.read_dataset import read_dataset
from src.prepare_dataset import prepare_dataset
from src.define_model import conv4

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold

def main():

# ─── Configurations
#
    print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))

    name = "pacs"
    imgs_path = 'data/' + name

# ─── Leitura do dataset
#
    X, y, groups = read_dataset(path=imgs_path, name=name)

# ─── Separation of holdout test (group-aware)
#
    unique_groups = np.unique(groups)

    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=0.2,
        random_state=42
    )

    train_mask = np.isin(groups, train_groups)
    test_mask  = np.isin(groups, test_groups)

    X_trainval, y_trainval = X[train_mask], y[train_mask]
    X_test,     y_test     = X[test_mask],  y[test_mask]

    groups_trainval = groups[train_mask]

    print(f"\nHoldout: {len(X_trainval)} imagens para treino/val | {len(X_test)} para teste")

# ─── K-5-Fold (group-aware by KFold on unique groups)
#
    unique_trainval_groups = np.unique(groups_trainval)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_g_idx, val_g_idx) in enumerate(kf.split(unique_trainval_groups)):

        # Grupos deste fold
        fold_train_groups = unique_trainval_groups[train_g_idx]
        fold_val_groups   = unique_trainval_groups[val_g_idx]

        # Mapeia grupos → imagens
        fold_train_mask = np.isin(groups_trainval, fold_train_groups)
        fold_val_mask   = np.isin(groups_trainval, fold_val_groups)

        X_train, y_train = X_trainval[fold_train_mask], y_trainval[fold_train_mask]
        X_val,   y_val   = X_trainval[fold_val_mask],   y_trainval[fold_val_mask]

        print(f"\nFold {fold+1}: {len(X_train)} treino | {len(X_val)} validação")

# ─── ...
#
    model = conv4()
    model.summary()

# ─── Final evaluation on holdout test
#
    print(f"\nAvaliação final em {len(X_test)} imagens de holdout test")
    # → model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
