import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import top_k_accuracy_score
import json

def cal_kl_divergence(y_true, y_pred, target_names=None, return_dict=False):
    """
    Calculate KL divergence between actual and predicted distributions.
    Can be used both as a scorer for model optimization and for evaluation.
    
    Args:
        y_true: Actual target values
        y_pred: Predicted target values
        target_names: Names of target columns (optional)
        return_dict: If True, returns a dictionary with KL scores for each target
                     If False, returns negative mean KL divergence for model optimization
    Returns:
        Negative mean KL divergence or dictionary with KL scores
    """
    # Convert DataFrame to numpy array if needed
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    
    # Get target column names
    if target_names is None and hasattr(y_true, 'columns'):
        target_names = y_true.columns
    elif target_names is None:
        target_names = [f"target_{i}" for i in range(y_true.shape[1])]
    
    # Calculate KL divergence for each target
    kl_scores = {}
    for i, target in enumerate(target_names):
        # Get actual and predicted distributions
        actual_counts = np.bincount(y_true[:, i].astype(int), minlength=np.max(y_true[:, i])+1)
        pred_counts = np.bincount(y_pred[:, i].astype(int), minlength=np.max(y_true[:, i])+1)
        
        # Ensure equal lengths
        max_len = max(len(actual_counts), len(pred_counts))
        actual_counts = np.pad(actual_counts, (0, max_len - len(actual_counts)), 'constant')
        pred_counts = np.pad(pred_counts, (0, max_len - len(pred_counts)), 'constant')
        
        # Convert to probability distributions
        smooth = 1e-10  # Smoothing to avoid zero probabilities
        actual_dist = actual_counts + smooth
        pred_dist = pred_counts + smooth
        
        # Normalize
        actual_dist = actual_dist / actual_dist.sum()
        pred_dist = pred_dist / pred_dist.sum()
        
        # Calculate KL divergence
        kl_div = entropy(actual_dist, pred_dist)
        kl_scores[target] = kl_div
    
    # Calculate overall average
    kl_scores['average'] = np.mean(list(kl_scores.values()))
    
    # Return appropriate value based on mode
    if return_dict:
        return kl_scores
    else:
        # for optimization
        return kl_scores['average']

def cal_group_kl_divergence(model=None, X_eval=None, y_eval=None, group_features=None, result_df=None):
    """
    Analyze KL divergence across different demographic or feature groups.
    
    Args:
        model: Trained model with predict method
        X_eval: Features for evaluation
        y_eval: Target values for evaluation
        group_features: List of feature names to group by
        result_df: DataFrame with predictions and features (will be created if None)
        
    Returns:
        DataFrame with KL divergence results by feature groups
    """
    if result_df is not None:
        target_cols = ['primary_mode', 'duration_minutes']
        pred_cols = ['predicted_mode','predicted_duration']
    elif isinstance(y_eval, pd.DataFrame):
        target_cols = y_eval.columns
        pred_cols = [f"predicted_{col}" for col in target_cols]
    else:
        target_cols = [f"target_{i}" for i in range(y_eval.shape[1])]
        pred_cols = [f"predicted_target_{i}" for i in range(y_eval.shape[1])]

    # Create result dataframe if not provided
    if result_df is None:
        # Get predictions
        y_pred = model.predict(X_eval)
        
        # Combine features, actual values and predictions
        if isinstance(X_eval, pd.DataFrame):
            result_df = X_eval.copy()
        else:
            result_df = pd.DataFrame(X_eval)
            
        # Add actual values
        if isinstance(y_eval, pd.DataFrame):
            for col in y_eval.columns:
                result_df[col] = y_eval[col].values
        else:
            for i in range(y_eval.shape[1]):
                result_df[f"target_{i}"] = y_eval[:, i]
                
        # Add predictions
        if isinstance(y_eval, pd.DataFrame):
            for i, col in enumerate(y_eval.columns):
                result_df[f"predicted_{col}"] = y_pred[:, i]
        else:
            for i in range(y_pred.shape[1]):
                result_df[f"predicted_target_{i}"] = y_pred[:, i]
    
    # Create a dataframe to store KL divergence results
    kl_results = []
    
    # Calculate KL divergence for each feature group
    for feature in group_features:
        # Get all unique values for this feature
        unique_values = result_df[feature].unique()
        
        for value in unique_values:
            # Filter data for this specific feature value
            group_df = result_df[result_df[feature] == value]
            
            # Skip groups with too few samples
            if len(group_df) < 20:
                continue
                
            for i, target in enumerate(target_cols):
                pred_col = pred_cols[i]
                # Calculate actual and predicted distributions
                actual_counts = group_df[target].value_counts(normalize=True)
                pred_counts = group_df[pred_col].value_counts(normalize=True)
                
                # Ensure distributions have the same indices
                all_values = sorted(set(actual_counts.index) | set(pred_counts.index))
                actual_dist = np.array([actual_counts.get(val, 0) for val in all_values])
                pred_dist = np.array([pred_counts.get(val, 0) for val in all_values])
                
                # Add smoothing to avoid zero probabilities
                smooth = 1e-10
                actual_dist = actual_dist + smooth
                pred_dist = pred_dist + smooth
                
                # Normalize
                actual_dist = actual_dist / actual_dist.sum()
                pred_dist = pred_dist / pred_dist.sum()
                
                # Calculate KL divergence
                kl_div = entropy(actual_dist, pred_dist)

                # Calculate mean absolute percentage error(mape)
                mape = np.mean(np.abs((actual_dist - pred_dist) / np.maximum(actual_dist, 1e-10)))
                mape = np.clip(mape, 0, 1)
                
                # Store results
                kl_results.append({
                    'feature': feature,
                    'value': value,
                    'target': target,
                    'sample_size': len(group_df),
                    'kl_divergence': kl_div,
                    'mape':mape
                })
    
    # Create results dataframe
    kl_df = pd.DataFrame(kl_results)
    
    # Calculate overall average KL divergence
    overall_kl = kl_df['kl_divergence'].mean()
    overall_mape = kl_df['mape'].mean()
    # print(f"Overall average KL divergence: {overall_kl:.4f}")
    # print(f"Overall mean absolute percentage error: {overall_mape:.4f}")
    return kl_df, overall_kl,overall_mape

def cal_topk_acc(model=None, X_eval=None, y_eval=None, result_df=None,k=3):
    results = {}
    target_cols=['primary_mode', 'duration_minutes']
    if result_df is not None:
        for col in target_cols:
            # Calculate mode accuracy
            y_true = result_df[col].values
            all_values = sorted(result_df[col].unique())
            value_to_idx = {val: idx for idx, val in enumerate(all_values)}
            
            # Create score matrix
            n_samples = len(result_df)
            n_classes = len(all_values)
            y_score = np.zeros((n_samples, n_classes))
            
            # Fill score matrix with weights from choice_weights
            for i, choice_weights_json in enumerate(result_df['choice_weights']):
                choices = json.loads(choice_weights_json)
                for choice in choices:
                    val = choice[col]
                    if val in value_to_idx:  # Skip values not in ground truth set
                        y_score[i, value_to_idx[val]] = choice['weight']
            
            # Convert ground truth to indices
            y_true_idx = np.array([value_to_idx.get(val, -1) for val in y_true])
            
            # Calculate top-k accuracy
            acc_score = top_k_accuracy_score(y_true_idx, y_score, k=k, labels=range(n_classes))
            results[col] = acc_score.item()
    else:
        y_pred_proba = model.predict_proba(X_eval)
        for i, col in enumerate(target_cols):
            y_true = y_eval[col].values
            unique_labels = np.unique(y_true)
            
            # Get the classes that the model knows about
            model_classes = model.classes_[i] if hasattr(model, 'classes_') and isinstance(model.classes_, list) else np.arange(y_pred_proba[i].shape[1])

            # Create converted y_true and expanded y_score
            n_samples = len(y_true)
            max_label = max(max(unique_labels), max(model_classes)) if len(model_classes) > 0 else max(unique_labels)
            n_classes = int(max_label) + 1
            
            # Expand prediction probabilities to include all classes in y_true
            expanded_proba = np.zeros((n_samples, n_classes))
            for j, label in enumerate(model_classes):
                if label < n_classes:  # Ensure label index is within bounds
                    expanded_proba[:, int(label)] = y_pred_proba[i][:, j]
            
            # Calculate top-k accuracy with all unique labels
            acc_score = top_k_accuracy_score(y_true, expanded_proba, k=k, labels=unique_labels)
            results[col] = acc_score
    # Calculate average accuracy across all target columns
    results['average'] = sum(results.values()) / len(results)
    # print(f"Top {k} accuracy: {results['average']:.4f}")
    return results

