# import numpy as np
# import json
# import os
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# def calculate_metrics(Y, Y_hat, output_name, norm_stats, output_dir):
#     col_names = norm_stats['columns']
#     target_idx = col_names.index(output_name)
#     target_mean = norm_stats['mean'][target_idx]
#     target_std = norm_stats['std'][target_idx]
    
#     # Denormalize both Y and Y_hat
#     Y_denorm = Y * target_std + target_mean
#     Y_hat_denorm = Y_hat * target_std + target_mean
    
#     # Get prediction horizon
#     H = Y.shape[1]
    
#     # Initialize metrics storage
#     metrics = {
#         'per_horizon': [],
#         'overall': {}
#     }
    
#     # Calculate metrics for each horizon
#     for h in range(H):
#         y_true = Y_denorm[:, h]
#         y_pred = Y_hat_denorm[:, h]
        
#         # Basic metrics
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         mae = mean_absolute_error(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)
        
#         # Percentage-based metrics
#         abs_perc_error = np.abs((y_true - y_pred) / y_true)
#         abs_sym_error = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
        
#         # Handle division by zero for percentage metrics
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mape = np.mean(np.where(y_true != 0, abs_perc_error, np.nan)) * 100
#             wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
#             smape = np.mean(np.where((y_true != 0) | (y_pred != 0), abs_sym_error, np.nan)) * 100
        
#         metrics['per_horizon'].append({
#             'horizon': h+1,
#             'RMSE': rmse,
#             'MAE': mae,
#             'R2': r2,
#             'MAPE': mape,
#             'WAPE': wape,
#             'SMAPE': smape
#         })
    
#     # Calculate overall metrics (average across horizons)
#     all_rmse = [m['RMSE'] for m in metrics['per_horizon']]
#     all_mae = [m['MAE'] for m in metrics['per_horizon']]
#     all_r2 = [m['R2'] for m in metrics['per_horizon']]
#     all_mape = [m['MAPE'] for m in metrics['per_horizon']]
#     all_wape = [m['WAPE'] for m in metrics['per_horizon']]
#     all_smape = [m['SMAPE'] for m in metrics['per_horizon']]
    
#     metrics['overall'] = {
#         'RMSE': np.mean(all_rmse),
#         'MAE': np.mean(all_mae),
#         'R2': np.mean(all_r2),
#         'MAPE': np.mean(all_mape),
#         'WAPE': np.mean(all_wape),
#         'SMAPE': np.mean(all_smape)
#     }
    
#     # Save results to JSON
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f'metrics_{output_name}.json')
#     with open(output_path, 'w') as f:
#         json.dump(metrics, f, indent=4)
    
#     return metrics


# import numpy as np
# import json
# import os
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# def calculate_metrics(Y, Y_hat, output_name, norm_stats, output_dir):
#     col_names = norm_stats['columns']
#     target_idx = col_names.index(output_name)
#     target_mean = norm_stats['mean'][target_idx]
#     target_std = norm_stats['std'][target_idx]
    
#     # Denormalize both Y and Y_hat
#     Y_denorm = Y * target_std + target_mean
#     Y_hat_denorm = Y_hat * target_std + target_mean
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Save true and predicted values as numpy files
#     np.save(os.path.join(output_dir, f'true_{output_name}.npy'), Y_denorm)
#     np.save(os.path.join(output_dir, f'pred_{output_name}.npy'), Y_hat_denorm)
    
#     # Get prediction horizon
#     H = Y.shape[1]
    
#     # Initialize metrics storage
#     metrics = {
#         'per_horizon': [],
#         'overall': {}
#     }
    
#     # Calculate metrics for each horizon
#     for h in range(H):
#         y_true = Y_denorm[:, h]
#         y_pred = Y_hat_denorm[:, h]
        
#         # Basic metrics
#         rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#         mae = mean_absolute_error(y_true, y_pred)
#         r2 = r2_score(y_true, y_pred)
        
#         # Percentage-based metrics
#         abs_perc_error = np.abs((y_true - y_pred) / (y_true+1e-15))
#         abs_sym_error = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
        
#         # Handle division by zero for percentage metrics
#         with np.errstate(divide='ignore', invalid='ignore'):
#             mape = np.mean(np.where(y_true != 0, abs_perc_error, np.nan)) * 100
#             wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
#             smape = np.mean(np.where((y_true != 0) | (y_pred != 0), abs_sym_error, np.nan)) * 100
        
#         metrics['per_horizon'].append({
#             'horizon': h+1,
#             'RMSE': rmse,
#             'MAE': mae,
#             'R2': r2,
#             'MAPE': mape,
#             'WAPE': wape,
#             'SMAPE': smape
#         })
    
#     # Calculate overall metrics (average across horizons)
#     all_rmse = [m['RMSE'] for m in metrics['per_horizon']]
#     all_mae = [m['MAE'] for m in metrics['per_horizon']]
#     all_r2 = [m['R2'] for m in metrics['per_horizon']]
#     all_mape = [m['MAPE'] for m in metrics['per_horizon']]
#     all_wape = [m['WAPE'] for m in metrics['per_horizon']]
#     all_smape = [m['SMAPE'] for m in metrics['per_horizon']]
    
#     metrics['overall'] = {
#         'RMSE': np.mean(all_rmse),
#         'MAE': np.mean(all_mae),
#         'R2': np.mean(all_r2),
#         'MAPE': np.mean(all_mape),
#         'WAPE': np.mean(all_wape),
#         'SMAPE': np.mean(all_smape)
#     }
    
#     # Save results to JSON
#     output_path = os.path.join(output_dir, f'metrics_{output_name}.json')
#     with open(output_path, 'w') as f:
#         json.dump(metrics, f, indent=4)
    
#     return metrics

import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(Y, Y_hat, output_name, norm_stats, output_dir):
    col_names = norm_stats['columns']
    target_idx = col_names.index(output_name)
    target_mean = norm_stats['mean'][target_idx]
    target_std = norm_stats['std'][target_idx]
    
    # Denormalize both Y and Y_hat
    Y_denorm = Y * target_std + target_mean
    Y_hat_denorm = Y_hat * target_std + target_mean
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save true and predicted values as numpy files
    np.save(os.path.join(output_dir, f'true_{output_name}.npy'), Y_denorm)
    np.save(os.path.join(output_dir, f'pred_{output_name}.npy'), Y_hat_denorm)
    
    # Get prediction horizon
    H = Y.shape[1]
    
    # Initialize metrics storage
    metrics = {
        'per_horizon': [],
        'overall': {}
    }
    
    # Calculate metrics for each horizon
    for h in range(H):
        y_true = Y_denorm[:, h]
        y_pred = Y_hat_denorm[:, h]
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate percentage-based metrics safely
        # Mask zero values to avoid division by zero
        mask = (y_true != 0) | (y_pred != 0)  # Handle cases where both are zero
        abs_perc_error = np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))
        abs_sym_error = 2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
        
        # Calculate metrics with safe masking
        mape = np.mean(abs_perc_error[mask]) * 100 if np.any(mask) else 0
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100 if np.sum(np.abs(y_true)) > 0 else 0
        smape = np.mean(abs_sym_error[mask]) * 100 if np.any(mask) else 0
        
        metrics['per_horizon'].append({
            'horizon': h+1,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'WAPE': wape,
            'SMAPE': smape
        })
    
    # Calculate overall metrics (average across horizons)
    all_rmse = [m['RMSE'] for m in metrics['per_horizon']]
    all_mae = [m['MAE'] for m in metrics['per_horizon']]
    all_r2 = [m['R2'] for m in metrics['per_horizon']]
    all_mape = [m['MAPE'] for m in metrics['per_horizon']]
    all_wape = [m['WAPE'] for m in metrics['per_horizon']]
    all_smape = [m['SMAPE'] for m in metrics['per_horizon']]
    
    metrics['overall'] = {
        'RMSE': np.mean(all_rmse),
        'MAE': np.mean(all_mae),
        'R2': np.mean(all_r2),
        'MAPE': np.mean(all_mape),
        'WAPE': np.mean(all_wape),
        'SMAPE': np.mean(all_smape)
    }
    
    # Save results to JSON
    output_path = os.path.join(output_dir, f'metrics_{output_name}.json')
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics