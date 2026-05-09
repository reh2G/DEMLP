import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.dataset.prepare_dataset import prepare_dataset
from src.dataset.strategies_dataset import apply_augmentation, apply_undersampling
from src.model.define_model import conv4
from src.model.utils_model import save_config, save_fold_groups, save_metrics, generate_gradcam, save_augmented_samples, save_excluded_images
from src.visual.plot_metrics import plot_average_learning_curves, plot_cumulative_confusion_matrix, plot_classification_report_bars, plot_roc_curve_cv, plot_pr_curve_cv, plot_metrics_boxplot
from src.visual.plot_errors import plot_worst_errors_gradcam

# ─── Run the complete experiment pipeline with all strategies and folds
#
def run_model(X, y, groups, config, output_dir):
    n_folds = config['n_folds']
    epochs = config['epochs']
    batch_size = config['batch_size']
    early_stopping_patience = config['early_stopping_patience']
    random_state = config['random_state']
    
    # Save configurations
    save_config(config, output_dir)
    
    # Separation of holdout test (group-aware)
    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=random_state)

    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)

    X_trainval, y_trainval = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    groups_trainval = groups[train_mask]

    print(f"Holdout: {len(X_trainval)} imagens para treino/val | {len(X_test)} para teste\n")

    strategies = ['default', 'augmentation', 'small-data']
    unique_trainval_groups = np.unique(groups_trainval)
    
    for strategy in strategies:
        print(f"\n{'='*50}\nIniciando Estratégia: {strategy.upper()}\n{'='*50}")
        
        strategy_dir = os.path.join(output_dir, strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Reset KFold for each strategy to ensure same splits
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Arrays to collect metrics across all folds
        histories = []
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        fold_metrics = []
        all_X_val = []
        
        for fold, (train_g_idx, val_g_idx) in enumerate(kf.split(unique_trainval_groups)):
            fold_dir = os.path.join(strategy_dir, f"fold_{fold+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            fold_train_groups = unique_trainval_groups[train_g_idx]
            fold_val_groups = unique_trainval_groups[val_g_idx]

            fold_train_mask = np.isin(groups_trainval, fold_train_groups)
            fold_val_mask = np.isin(groups_trainval, fold_val_groups)

            X_train, y_train = X_trainval[fold_train_mask], y_trainval[fold_train_mask]
            X_val, y_val = X_trainval[fold_val_mask], y_trainval[fold_val_mask]
            
            print(f"\n-- Fold {fold+1} --")
            print(f"Original: {len(X_train)} treino | {len(X_val)} validação")

            # Prepare the train and validation data (Normalization first!)
            X_train_prep, y_train_prep, X_val_prep, y_val_prep = prepare_dataset(X_train, y_train, X_val, y_val)
            
            # Apply the data strategy only for the train data (After normalization)
            if strategy == 'augmentation':
                extra_aug = config.get('extra_augment', 500)
                X_train_prep, y_train_prep, augmented_X = apply_augmentation(X_train_prep, y_train_prep, minority_class=1, majority_class=0, random_state=random_state, extra_augment=extra_aug)
                # Save augmented samples only on the first fold
                if fold == 0 and len(augmented_X) > 0:
                    # augmented_X needs to be scaled back to 0-255 to save image properly
                    save_augmented_samples((augmented_X * 255).astype(np.uint8), strategy_dir)
                    
            elif strategy == 'small-data':
                X_train_prep, y_train_prep, excluded_indices = apply_undersampling(X_train_prep, y_train_prep, minority_class=1, majority_class=0, random_state=random_state)
                # Save excluded images only on the first fold
                if fold == 0 and len(excluded_indices) > 0:
                    save_excluded_images(excluded_indices, X_trainval, y_trainval, fold_train_mask, strategy_dir)
                    
            print(f"Após estratégia '{strategy}': {len(X_train_prep)} treino | {len(X_val_prep)} validação\n")
            
            # Instantiate the model
            tf.keras.backend.clear_session()
            model = conv4()
            
            # Train the model with EarlyStopping
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True, verbose=1)
            history = model.fit(X_train_prep, y_train_prep, validation_data=(X_val_prep, y_val_prep), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
            
            # Evaluate the model
            y_pred_prob = model.predict(X_val_prep)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_val_prep, axis=1)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print(f"Métricas Fold {fold+1} - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            
            # Save metrics
            save_metrics(fold_dir, fold, acc, prec, rec, f1, y_true, y_pred)
            
            # Append fold results for strategy visualization
            histories.append(history.history)
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_prob.append(y_pred_prob[:, 1])
            fold_metrics.append({'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})
            all_X_val.append(X_val_prep)
            
            # Save the groups for this fold (complete, no truncation)
            save_fold_groups(fold_dir, fold_train_groups, fold_val_groups, test_groups)
            
            # Save the model
            model.save(os.path.join(fold_dir, 'model.keras'))
            
            # Grad-CAM: 3 healthy + 3 stone from validation
            generate_gradcam(model, X_val_prep, y_val_prep, fold_dir)
                
        print(f"\nFinalizada a validação cruzada da estratégia {strategy}.")
        print("Gerando visualizações globais...")
        
        plot_average_learning_curves(histories, strategy_dir)
        plot_cumulative_confusion_matrix(all_y_true, all_y_pred, strategy_dir)
        plot_classification_report_bars(all_y_true, all_y_pred, strategy_dir)
        plot_roc_curve_cv(all_y_true, all_y_prob, strategy_dir)
        plot_pr_curve_cv(all_y_true, all_y_prob, strategy_dir)
        plot_metrics_boxplot(fold_metrics, strategy_dir)
        
        print("Gerando Grad-CAM para os piores erros...")
        plot_worst_errors_gradcam(all_X_val, all_y_true, all_y_pred, all_y_prob, strategy_dir)
        
        print(f"[{strategy.upper()}] Visualizações salvas com sucesso.")
