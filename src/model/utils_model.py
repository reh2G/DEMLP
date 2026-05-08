import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.visual.grad_cam import make_gradcam_heatmap, save_and_display_gradcam

CLASS_NAMES = {0: 'Healthy', 1: 'Stone'}

# ─── Save experiment configurations to a text file
#
def save_config(config, output_dir):
    config_path = os.path.join(output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("═" * 40 + "\n")
        f.write("  Configurações do Experimento\n")
        f.write("═" * 40 + "\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"Configurações salvas em {config_path}")

# ─── Save complete group lists for each fold
#
def save_fold_groups(fold_dir, fold_train_groups, fold_val_groups, test_groups):
    with open(os.path.join(fold_dir, 'groups.txt'), 'w') as f:
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            f.write(f"Train Groups: {fold_train_groups}\n")
            f.write(f"Validation Groups: {fold_val_groups}\n")
            f.write(f"Test Groups: {test_groups}\n")

# ─── Save evaluation metrics for a fold
#
def save_metrics(fold_dir, fold, acc, prec, rec, f1, y_true, y_pred):
    with open(os.path.join(fold_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}\n")

# ─── Generate Grad-CAM for 3 random healthy and 3 random stone images
#
def generate_gradcam(model, X_val_prep, y_val_prep, fold_dir):
    y_true = np.argmax(y_val_prep, axis=1)
    
    for class_label, class_name in CLASS_NAMES.items():
        class_indices = np.where(y_true == class_label)[0]
        
        num_cams = min(3, len(class_indices))
        if num_cams == 0:
            continue
            
        selected = np.random.choice(class_indices, num_cams, replace=False)
        
        for i, idx in enumerate(selected):
            img_val = X_val_prep[idx]
            img_array = np.expand_dims(img_val, axis=0)
            
            # Get model prediction
            preds = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(preds[0])
            confidence = preds[0][predicted_class]
            true_class = y_true[idx]
            
            heatmap = make_gradcam_heatmap(img_array, model, "last_conv")
            save_and_display_gradcam(
                img_val, heatmap,
                cam_path=os.path.join(fold_dir, f"gradcam_{class_name.lower()}_{i+1}.jpg"),
                confidence=confidence,
                predicted_label=predicted_class,
                true_label=true_class
            )

# ─── Save 3 random augmented images for visualization
#
def save_augmented_samples(augmented_X, strategy_dir):
    samples_dir = os.path.join(strategy_dir, 'augmented_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    num_samples = min(3, len(augmented_X))
    if num_samples == 0:
        return
        
    selected = np.random.choice(len(augmented_X), num_samples, replace=False)
    
    for i, idx in enumerate(selected):
        img = augmented_X[idx]
        if np.max(img) <= 1.0:
            img = np.uint8(255 * img)
        else:
            img = np.uint8(img)
        
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Augmented Sample {i+1}', fontsize=11, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(samples_dir, f'augmented_{i+1}.jpg'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  {num_samples} amostras augmentadas salvas em {samples_dir}")

# ─── Save a text file listing the excluded images from undersampling
#
def save_excluded_images(excluded_indices, X_trainval, y_trainval, fold_train_mask, strategy_dir):
    # The excluded_indices are indices into the majority class subset within the fold training data
    X_train_fold = X_trainval[fold_train_mask]
    y_train_fold = y_trainval[fold_train_mask]
    
    # Get majority class (healthy = 0) indices within the fold train data
    majority_mask = (y_train_fold == 0)
    majority_global_indices = np.where(majority_mask)[0]
    
    excluded_path = os.path.join(strategy_dir, 'excluded_images.txt')
    with open(excluded_path, 'w') as f:
        f.write(f"Total de imagens excluídas: {len(excluded_indices)}\n")
        f.write(f"Classe: {CLASS_NAMES[0]} (majority)\n\n")
        for i, exc_idx in enumerate(excluded_indices):
            global_idx = majority_global_indices[exc_idx]
            f.write(f"  [{i+1}] Índice no fold: {global_idx}\n")
    
    print(f"  {len(excluded_indices)} imagens excluídas registradas em {excluded_path}")
