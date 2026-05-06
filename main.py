import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from src.read_dataset import read_dataset
from src.prepare_dataset import prepare_dataset
from src.define_model import conv4
from src.data_strategies import apply_augmentation, apply_undersampling
from src.grad_cam import make_gradcam_heatmap, save_and_display_gradcam

def main():
    # ─── Configurations
    print("GPUs disponíveis: ", tf.config.list_physical_devices('GPU'))
    DEBUG = False
    SIMILARITY = 0.75
    name = "combined"
    imgs_path = 'data/' + name

    # ─── Leitura do dataset
    # Se o dataset real não estiver na pasta 'data/combined' no momento do teste,
    # certifique-se de que ele esteja lá, ou crie diretórios fictícios para teste se necessário.
    X, y, groups = read_dataset(path=imgs_path, name=name, DEBUG=DEBUG, SIMILARITY=SIMILARITY)

    # ─── Separation of holdout test (group-aware)
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=53)

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    X_trainval, y_trainval = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups_trainval = groups[train_mask]

    print(f"Holdout: {len(X_trainval)} imagens para treino/val | {len(X_test)} para teste\n")

    strategies = ['padrao', 'augmentation', 'small-data']
    
    unique_trainval_groups = np.unique(groups_trainval)
    
    for strategy in strategies:
        print(f"\n{'='*50}\nIniciando Estratégia: {strategy.upper()}\n{'='*50}")
        
        # Reset KFold for each strategy to ensure same splits
        kf = KFold(n_splits=5, shuffle=True, random_state=53)
        
        for fold, (train_g_idx, val_g_idx) in enumerate(kf.split(unique_trainval_groups)):
            fold_dir = f"output/{strategy}/fold_{fold+1}"
            os.makedirs(fold_dir, exist_ok=True)
            
            fold_train_groups = unique_trainval_groups[train_g_idx]
            fold_val_groups = unique_trainval_groups[val_g_idx]

            fold_train_mask = np.isin(groups_trainval, fold_train_groups)
            fold_val_mask = np.isin(groups_trainval, fold_val_groups)

            X_train, y_train = X_trainval[fold_train_mask], y_trainval[fold_train_mask]
            X_val, y_val = X_trainval[fold_val_mask], y_trainval[fold_val_mask]
            
            print(f"\n-- Fold {fold+1} --")
            print(f"Original: {len(X_train)} treino | {len(X_val)} validação")

            # Aplica a estratégia de dados
            if strategy == 'augmentation':
                X_train, y_train = apply_augmentation(X_train, y_train, minority_class=1, majority_class=0)
            elif strategy == 'small-data':
                X_train, y_train = apply_undersampling(X_train, y_train, minority_class=1, majority_class=0)
                
            print(f"Após estratégia '{strategy}': {len(X_train)} treino")
            
            # Prepara dados (normalização + one-hot)
            X_train_prep, y_train_prep, X_val_prep, y_val_prep = prepare_dataset(X_train, y_train, X_val, y_val)
            
            # Instancia o modelo
            tf.keras.backend.clear_session()
            model = conv4()
            
            # Treina o modelo (Usando apenas 5 épocas como padrão para não demorar muito, altere conforme necessário)
            history = model.fit(
                X_train_prep, y_train_prep,
                validation_data=(X_val_prep, y_val_prep),
                epochs=5,
                batch_size=32,
                verbose=1
            )
            
            # Avalia o modelo
            y_pred_prob = model.predict(X_val_prep)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_val_prep, axis=1)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print(f"Métricas Fold {fold+1} - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            
            # Salvar métricas
            with open(f"{fold_dir}/metrics.txt", "w") as f:
                f.write(f"Accuracy: {acc:.4f}\n")
                f.write(f"Precision: {prec:.4f}\n")
                f.write(f"Recall: {rec:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")
            
            # Salvar os grupos que caíram neste fold para documentação
            with open(f"{fold_dir}/groups.txt", "w") as f:
                f.write(f"Train Groups: {fold_train_groups}\n")
                f.write(f"Validation Groups: {fold_val_groups}\n")
            
            # Salvar o modelo
            model.save(f"{fold_dir}/model.keras")
            
            # Grad-CAM para 3 imagens aleatórias de validação
            num_cams = min(3, len(X_val_prep))
            cam_indices = np.random.choice(len(X_val_prep), num_cams, replace=False)
            
            for i, idx in enumerate(cam_indices):
                img_val = X_val_prep[idx]
                img_array = np.expand_dims(img_val, axis=0) # shape (1, H, W, C)
                
                heatmap = make_gradcam_heatmap(img_array, model, "last_conv")
                save_and_display_gradcam(img_val, heatmap, cam_path=f"{fold_dir}/gradcam_val_{i+1}.jpg")
                
        print(f"\nFinalizada a estratégia {strategy}")

if __name__ == "__main__":
    main()
