import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

CLASS_NAMES = ['Healthy', 'Stone']

# ─── Plot average learning curves across folds
#
def plot_average_learning_curves(histories, output_dir):
    if not histories:
        return

    # Assuming all folds ran for the same number of epochs, or pad to max epochs
    max_epochs = max(len(h['loss']) for h in histories)
    
    # Initialize arrays
    loss = np.zeros((len(histories), max_epochs))
    val_loss = np.zeros((len(histories), max_epochs))
    acc = np.zeros((len(histories), max_epochs))
    val_acc = np.zeros((len(histories), max_epochs))
    
    for i, h in enumerate(histories):
        e = len(h['loss'])
        loss[i, :e] = h['loss']
        val_loss[i, :e] = h['val_loss']
        acc[i, :e] = h.get('accuracy', h.get('acc', []))
        val_acc[i, :e] = h.get('val_accuracy', h.get('val_acc', []))
        
        # If early stopping occurred, forward-fill the last value
        if e < max_epochs:
            loss[i, e:] = h['loss'][-1]
            val_loss[i, e:] = h['val_loss'][-1]
            acc[i, e:] = h.get('accuracy', h.get('acc', [-1]))[-1]
            val_acc[i, e:] = h.get('val_accuracy', h.get('val_acc', [-1]))[-1]

    epochs = np.arange(1, max_epochs + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(epochs, np.mean(loss, axis=0), 'b-', label='Train Loss (Mean)')
    axes[0].plot(epochs, np.mean(val_loss, axis=0), 'r-', label='Val Loss (Mean)')
    axes[0].fill_between(epochs, np.mean(loss, axis=0) - np.std(loss, axis=0), np.mean(loss, axis=0) + np.std(loss, axis=0), color='b', alpha=0.1)
    axes[0].fill_between(epochs, np.mean(val_loss, axis=0) - np.std(val_loss, axis=0), np.mean(val_loss, axis=0) + np.std(val_loss, axis=0), color='r', alpha=0.1)
    axes[0].set_title('Average Learning Curves (Loss)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Accuracy
    axes[1].plot(epochs, np.mean(acc, axis=0), 'b-', label='Train Acc (Mean)')
    axes[1].plot(epochs, np.mean(val_acc, axis=0), 'r-', label='Val Acc (Mean)')
    axes[1].fill_between(epochs, np.mean(acc, axis=0) - np.std(acc, axis=0), np.mean(acc, axis=0) + np.std(acc, axis=0), color='b', alpha=0.1)
    axes[1].fill_between(epochs, np.mean(val_acc, axis=0) - np.std(val_acc, axis=0), np.mean(val_acc, axis=0) + np.std(val_acc, axis=0), color='r', alpha=0.1)
    axes[1].set_title('Average Learning Curves (Accuracy)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Plot cumulative confusion matrix
#
def plot_cumulative_confusion_matrix(all_y_true, all_y_pred, output_dir):
    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)
    
    cm = confusion_matrix(y_true_flat, y_pred_flat)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Cumulative Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Plot classification report as a grouped horizontal bar chart
#
def plot_classification_report_bars(all_y_true, all_y_pred, output_dir):
    import pandas as pd

    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)

    report = classification_report(y_true_flat, y_pred_flat, target_names=CLASS_NAMES, output_dict=True)

    # Extract per-class and averages (excluding 'accuracy' row)
    rows = []
    row_labels = CLASS_NAMES + ['macro avg', 'weighted avg']
    for label in row_labels:
        if label in report:
            rows.append({
                'Metrics': label,
                'Precision': report[label]['precision'],
                'Recall':    report[label]['recall'],
                'F1-Score':  report[label]['f1-score'],
            })

    df = pd.DataFrame(rows)
    df_melted = df.melt(id_vars=['Metrics'], value_vars=['Precision', 'Recall', 'F1-Score'],
                        var_name='Metric', value_name='Score')

    # Overall accuracy scalar
    accuracy = report.get('accuracy', None)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_melted, y='Metrics', x='Score', hue='Metric', palette='muted', ax=ax)

    # Annotate bar values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Score')
    ax.set_ylabel('')
    ax.set_title('Classification Report')

    # Draw accuracy as a vertical dashed line
    if accuracy is not None:
        ax.axvline(x=accuracy, color='crimson', linestyle='--', linewidth=1.5,
                   label=f'Accuracy = {accuracy:.4f}')

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_report.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Plot ROC Curve with Confidence Interval
#
def plot_roc_curve_cv(all_y_true, all_y_prob, output_dir):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for i, (y_true, y_prob) in enumerate(zip(all_y_true, all_y_prob)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # Interpolate TPR for plotting the mean curve
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        ax.plot(fpr, tpr, color='gray', alpha=0.3, lw=1)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b', label=rf'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})', lw=2, alpha=0.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='b', alpha=0.2, label=r'$\pm$ 1 std. dev.')
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Plot Precision-Recall Curve
#
def plot_pr_curve_cv(all_y_true, all_y_prob, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    
    y_true_flat = np.concatenate(all_y_true)
    y_prob_flat = np.concatenate(all_y_prob)
    
    precision, recall, _ = precision_recall_curve(y_true_flat, y_prob_flat)
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, color='b', label=f'PR Curve (AUC = {pr_auc:.3f})', lw=2)
    
    # Calculate baseline
    no_skill = len(y_true_flat[y_true_flat==1]) / len(y_true_flat)
    ax.plot([0, 1], [no_skill, no_skill], linestyle='--', color='r', label='No Skill')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Plot metrics distribution (Boxplot)
#
def plot_metrics_boxplot(fold_metrics, output_dir):
    import pandas as pd
    df = pd.DataFrame(fold_metrics)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, ax=ax, palette="Set2")
    sns.swarmplot(data=df, color=".25", size=6, ax=ax)
    
    ax.set_title('Metrics Distribution Across Folds')
    ax.set_ylabel('Score')
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
