import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from src.visual.grad_cam import make_gradcam_heatmap, CLASS_NAMES

# ─── Plot the worst errors with Grad-CAM overlay
#
def plot_worst_errors_gradcam(all_X_val, all_y_true, all_y_pred, all_y_prob, strategy_dir):
    fp_records = []
    fn_records = []
    
    for fold_idx in range(len(all_y_true)):
        y_true = all_y_true[fold_idx]
        y_pred = all_y_pred[fold_idx]
        y_prob = all_y_prob[fold_idx] # Prob of class 1 (Stone)
        
        for i in range(len(y_true)):
            if y_true[i] == 0 and y_pred[i] == 1:
                # False Positive: true is Healthy (0), pred is Stone (1)
                # Confidence in Stone is high (close to 1)
                fp_records.append({
                    'fold_idx': fold_idx,
                    'img_idx': i,
                    'prob': y_prob[i]
                })
            elif y_true[i] == 1 and y_pred[i] == 0:
                # False Negative: true is Stone (1), pred is Healthy (0)
                # Confidence in Healthy is high (prob of Stone is close to 0)
                fn_records.append({
                    'fold_idx': fold_idx,
                    'img_idx': i,
                    'prob': y_prob[i]
                })
                
    # Sort FP descending by prob (worst errors have prob close to 1)
    fp_records.sort(key=lambda x: x['prob'], reverse=True)
    # Sort FN ascending by prob (worst errors have prob close to 0)
    fn_records.sort(key=lambda x: x['prob'])
    
    top_fp = fp_records[:3]
    top_fn = fn_records[:3]
    
    generate_and_save_error_panel(top_fp, "False Positives (Predicted Stone, True Healthy)", "fp", all_X_val, strategy_dir)
    generate_and_save_error_panel(top_fn, "False Negatives (Predicted Healthy, True Stone)", "fn", all_X_val, strategy_dir)

# ─── Generate and save the grid image of the worst errors
#
def generate_and_save_error_panel(records, title, prefix, all_X_val, strategy_dir):
    if not records:
        print(f"No {prefix} errors found.")
        return
        
    num_errors = len(records)
    fig, axes = plt.subplots(num_errors, 2, figsize=(8, 4 * num_errors))
    
    # If only 1 error, axes is 1D array. Make it 2D for consistency
    if num_errors == 1:
        axes = np.expand_dims(axes, axis=0)
        
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    loaded_models = {}
    
    for row, record in enumerate(records):
        fold_idx = record['fold_idx']
        img_idx = record['img_idx']
        prob = record['prob']
        
        # Load model if not loaded
        if fold_idx not in loaded_models:
            model_path = os.path.join(strategy_dir, f"fold_{fold_idx+1}", 'model.keras')
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}. Skipping.")
                continue
            loaded_models[fold_idx] = tf.keras.models.load_model(model_path)
            
        model = loaded_models[fold_idx]
        img_val = all_X_val[fold_idx][img_idx]
        img_array = np.expand_dims(img_val, axis=0)
        
        heatmap = make_gradcam_heatmap(img_array, model, "last_conv")
        
        # Resize heatmap
        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (img_val.shape[0], img_val.shape[1])).numpy()[..., 0]
        
        # Original Image
        ax_orig = axes[row, 0]
        if np.max(img_val) <= 1.0:
            ax_orig.imshow(img_val, cmap='gray')
        else:
            ax_orig.imshow(img_val.astype('uint8'), cmap='gray')
        ax_orig.set_title(f"Fold {fold_idx+1}\nProb(Stone)={prob:.3f}", fontsize=10)
        ax_orig.axis('off')
        
        # Grad-CAM overlay
        ax_cam = axes[row, 1]
        if np.max(img_val) <= 1.0:
            ax_cam.imshow(img_val, cmap='gray')
        else:
            ax_cam.imshow(img_val.astype('uint8'), cmap='gray')
        ax_cam.imshow(heatmap_resized, cmap='jet', alpha=0.4)
        ax_cam.set_title("Grad-CAM Focus", fontsize=10)
        ax_cam.axis('off')
        
    plt.tight_layout()
    # Adjust layout to make room for suptitle
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(strategy_dir, f'worst_errors_{prefix}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Clear session to free memory
    tf.keras.backend.clear_session()
    loaded_models.clear()
