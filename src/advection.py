


# import numpy as np
# from scipy.optimize import minimize
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # Fixed physical constants
# DX = 100000  # 100 km grid spacing in meters
# DT = 3600    # 1 hour in seconds

# # Initial parameters for optimization
# INIT_ALPHA = 5e-5  # Initial thermal diffusivity
# INIT_BETA = 1.0     # Scaling factor for update magnitude

# def parameterized_physics_model(X, norm_stats, col_map, H, alpha, beta):
#     """Physics-based temperature prediction with tunable parameters"""
#     # Create column name to index mapping
#     col_to_idx = {col: idx for idx, col in col_map.items()}
    
#     # Get normalization parameters
#     means = np.array(norm_stats['mean'])
#     stds = np.array(norm_stats['std'])
    
#     X_denorm = X 
#     # Initialize prediction array
#     num_samples = X.shape[0]
#     Y_phy = np.zeros((num_samples, H))
    
#     # Precompute indices for all required features
#     center_t2m_idx = col_to_idx['C_T2M']
#     center_ws_idx = col_to_idx['C_WS50M']
#     center_wd_idx = col_to_idx['C_WD50M']
    
#     # Neighbor directions and their indices
#     directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
#     neighbor_indices = {dir: col_to_idx[f'{dir}_T2M'] for dir in directions}
    
#     # Get last timestep for all samples
#     last_steps = X_denorm[:, -1, :]
    
#     # Extract center data
#     T_c = last_steps[:, center_t2m_idx]
#     ws = last_steps[:, center_ws_idx]
#     wd = last_steps[:, center_wd_idx]
    
#     # Extract neighbor temperatures
#     neighbors = {}
#     for dir, idx in neighbor_indices.items():
#         neighbors[dir] = last_steps[:, idx]
    
#     # Convert wind direction to radians
#     wd_rad = np.radians(wd)
    
#     # Compute wind components
#     u = -ws * np.sin(wd_rad)  # East-west component
#     v = -ws * np.cos(wd_rad)  # North-south component
    
#     # Precompute constant terms
#     dx_factor = 1 / (2 * DX)
#     diff_const = alpha / (DX**2)
    
#     # Predict H steps using physics model
#     current_temp = T_c.copy()
#     for h in range(H):
#         # Calculate temperature gradients
#         dT_dx = (neighbors['E'] - neighbors['W']) * dx_factor
#         dT_dy = (neighbors['N'] - neighbors['S']) * dx_factor
        
#         # Calculate advection term
#         advection = -(u * dT_dx + v * dT_dy)
        
#         # Calculate diffusion term (9-point stencil)
#         cardinals = (neighbors['N'] + neighbors['E'] + 
#                      neighbors['S'] + neighbors['W'] - 4 * current_temp)
#         diagonals = (neighbors['NE'] + neighbors['SE'] + 
#                      neighbors['SW'] + neighbors['NW'] - 4 * current_temp)
        
#         diffusion = diff_const * (4/6 * cardinals + 1/12 * diagonals)
        
#         # Update temperature with scaling factor
#         current_temp += beta * (advection + diffusion) * DT
#         Y_phy[:, h] = current_temp
    
#     # Normalize predictions using C_T2M stats
#     c_t2m_mean = means[center_t2m_idx]
#     c_t2m_std = stds[center_t2m_idx]
    
#     return Y_phy

# def physics_loss(params, X, Y_true, norm_stats, col_map, H):
#     """Custom loss function: (RMSE + MAE) / R2_score"""
#     alpha, beta = params
    
#     try:
#         Y_pred = parameterized_physics_model(X, norm_stats, col_map, H, alpha, beta)
        
#         # Flatten predictions and truth
#         Y_true_flat = Y_true.flatten()
#         Y_pred_flat = Y_pred.flatten()
        
#         # Remove NaN/inf values
#         valid_mask = np.isfinite(Y_true_flat) & np.isfinite(Y_pred_flat)
#         Y_true_flat = Y_true_flat[valid_mask]
#         Y_pred_flat = Y_pred_flat[valid_mask]
        
#         # Calculate metrics
#         rmse = np.sqrt(mean_squared_error(Y_true_flat, Y_pred_flat))
#         mae = mean_absolute_error(Y_true_flat, Y_pred_flat)
#         r2 = r2_score(Y_true_flat, Y_pred_flat)
        
#         # Handle negative R²
#         if r2 <= 0:
#             return 1000 * (rmse + mae)  # Large penalty
        
#         # Custom loss: (RMSE + MAE) / R2
#         return (rmse + mae) / r2
        
#     except Exception as e:
#         print(f"Error in physics_loss: {e}")
#         return float('inf')

# def optimize_physics_model(X_train, Y_train, norm_stats, col_map, H):
#     """Optimize physics model parameters to minimize loss on training set"""
#     # Bounds for parameters
#     bounds = [(1e-7, 1e-3), (0.01, 5.0)]  # Wider bounds
    
#     # Initial parameters
#     initial_params = [INIT_ALPHA, INIT_BETA]
    
#     # Use a subset for optimization if dataset is large
#     if X_train.shape[0] > 1000:
#         idx = np.random.choice(X_train.shape[0], 1000, replace=False)
#         X_sub = X_train[idx]
#         Y_sub = Y_train[idx]
#     else:
#         X_sub = X_train
#         Y_sub = Y_train
    
#     try:
#         # Use differential evolution for better global optimization
#         result = minimize(physics_loss, initial_params, 
#                          args=(X_sub, Y_sub, norm_stats, col_map, H),
#                          method='Nelder-Mead',  # More robust to noise
#                          options={'maxiter': 20, 'disp': True, 'xatol': 1e-6})
        
#         if result.success:
#             print(f"Optimization succeeded with loss: {result.fun:.4f}")
#             print(f"Optimal alpha: {result.x[0]:.2e}")
#             print(f"Optimal beta: {result.x[1]:.4f}")
#             return result.x
#         else:
#             print("Optimization did not converge, using initial parameters")
#             return initial_params
            
#     except Exception as e:
#         print(f"Optimization failed: {e}")
#         return initial_params

# def get_optimized_physics_predictions(X_train, Y_train, X_val, X_test, 
#                                      norm_stats, col_map, H):
#     """Get physics-based predictions with optimized parameters"""
#     # Step 1: Optimize parameters
#     opt_params = optimize_physics_model(X_train, Y_train, norm_stats, col_map, H)
#     alpha_opt, beta_opt = opt_params
    
#     # Step 2: Generate predictions
#     Y_train_phy = parameterized_physics_model(X_train, norm_stats, col_map, H, alpha_opt, beta_opt)
#     Y_val_phy = parameterized_physics_model(X_val, norm_stats, col_map, H, alpha_opt, beta_opt)
#     Y_test_phy = parameterized_physics_model(X_test, norm_stats, col_map, H, alpha_opt, beta_opt)
    
#     return Y_train_phy, Y_val_phy, Y_test_phy, opt_params

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Physical constants
DX = 100000  # 100 km grid spacing [m]
DT = 3600    # 1 hour [s]
MAX_WIND_SPEED = DX / DT  # CFL stability limit (~27.8 m/s)

def parameterized_physics_model(X, norm_stats, col_map, H, alpha, beta):
    """Physics-based prediction with proper denormalization and stability fixes"""
    # Create column name to index mapping
    col_to_idx = {col: idx for idx, col in col_map.items()}
    
    # Get normalization parameters
    means = np.array(norm_stats['mean'])
    stds = np.array(norm_stats['std'])
    
    # Proper denormalization
    X_denorm = X * stds + means
    
    # Initialize prediction array
    num_samples = X.shape[0]
    Y_phy = np.zeros((num_samples, H))
    
    # Precompute indices
    center_t2m_idx = col_to_idx['C_T2M']
    center_ws_idx = col_to_idx['C_WS50M']
    center_wd_idx = col_to_idx['C_WD50M']
    
    # Neighbor indices
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    neighbor_indices = {dir: col_to_idx[f'{dir}_T2M'] for dir in directions}
    
    # Get last timestep for all samples
    last_steps = X_denorm[:, -1, :]
    
    # Extract center data
    T_c = last_steps[:, center_t2m_idx]
    ws = last_steps[:, center_ws_idx]
    wd = last_steps[:, center_wd_idx]
    
    # Extract neighbor temperatures
    neighbors = {}
    for dir, idx in neighbor_indices.items():
        neighbors[dir] = last_steps[:, idx]
    
    # Convert wind direction to radians and clamp speed
    wd_rad = np.radians(wd)
    ws = np.clip(ws, 0, MAX_WIND_SPEED)  # Stability fix
    
    # Compute wind components
    u = -ws * np.sin(wd_rad)  # East-west
    v = -ws * np.cos(wd_rad)  # North-south
    
    # Precompute constants
    dx_factor = 1 / (2 * DX)
    diff_const = alpha / (DX**2)
    
    # Predict H steps
    current_temp = T_c.copy()
    neighbor_states = {k: v.copy() for k, v in neighbors.items()}
    
    for h in range(H):
        # Calculate temperature gradients
        dT_dx = (neighbor_states['E'] - neighbor_states['W']) * dx_factor
        dT_dy = (neighbor_states['N'] - neighbor_states['S']) * dx_factor
        
        # Advection term (with time step splitting)
        advection = np.zeros_like(current_temp)
        for _ in range(4):  # Sub-time stepping
            advection_term = -(u * dT_dx + v * dT_dy)
            current_temp += beta * advection_term * (DT/4)
            advection += advection_term
        
        # Diffusion term (9-point stencil)
        cardinals = (neighbor_states['N'] + neighbor_states['E'] + 
                     neighbor_states['S'] + neighbor_states['W'] - 4 * current_temp)
        diagonals = (neighbor_states['NE'] + neighbor_states['SE'] + 
                     neighbor_states['SW'] + neighbor_states['NW'] - 4 * current_temp)
        diffusion = diff_const * (4/6 * cardinals + 1/12 * diagonals)
        
        # Update temperature
        current_temp += beta * diffusion * DT
        
        # Simple neighbor state propagation (50% center influence)
        for dir in directions:
            neighbor_states[dir] += 0.5 * (current_temp - neighbor_states[dir])
        
        Y_phy[:, h] = current_temp
    
    # Renormalize predictions
    c_t2m_mean = means[center_t2m_idx]
    c_t2m_std = stds[center_t2m_idx]
    return (Y_phy - c_t2m_mean) / c_t2m_std

def physics_loss(params, X, Y_true, norm_stats, col_map, H):
    """Stable loss function with physical constraints"""
    alpha, beta = params
    
    try:
        Y_pred = parameterized_physics_model(X, norm_stats, col_map, H, alpha, beta)
        Y_true_flat = Y_true.flatten()
        Y_pred_flat = Y_pred.flatten()
        
        # Validity mask
        valid_mask = np.isfinite(Y_true_flat) & np.isfinite(Y_pred_flat)
        if np.sum(valid_mask) < 0.5 * len(Y_true_flat):
            return 1e10  # Large penalty for mostly invalid results
        
        Y_true_flat = Y_true_flat[valid_mask]
        Y_pred_flat = Y_pred_flat[valid_mask]
        
        # Base RMSE
        rmse = np.sqrt(mean_squared_error(Y_true_flat, Y_pred_flat))
        
        # Physical constraint penalties
        temp_range = np.max(Y_true_flat) - np.min(Y_true_flat)
        temp_change = np.mean(np.abs(Y_pred_flat - Y_true_flat))
        penalty = 0.0
        
        # Penalize unrealistic parameter values
        if alpha < 1e-9 or alpha > 1e-2:
            penalty += 10 * rmse
        if beta < 0.1 or beta > 3.0:
            penalty += 10 * rmse
            
        # Penalize excessive temperature changes
        if temp_change > 2 * temp_range:
            penalty += 5 * rmse
            
        return rmse + penalty
        
    except Exception as e:
        print(f"Loss error: {e}")
        return 1e10

def optimize_physics_model(X_train, Y_train, norm_stats, col_map, H):
    """Robust parameter optimization using global method"""
    # Physical parameter bounds (log scale for alpha)
    bounds = [
        (1e-9, 1e-2),  # Alpha (thermal diffusivity)
        (0.1, 3.0)      # Beta (update scaling)
    ]
    
    # Use representative subset
    n_samples = min(2000, X_train.shape[0])
    idx = np.random.choice(X_train.shape[0], n_samples, replace=False)
    X_sub, Y_sub = X_train[idx], Y_train[idx]
    
    try:
        # Global optimization with population-based method
        result = differential_evolution(
            physics_loss,
            bounds,
            args=(X_sub, Y_sub, norm_stats, col_map, H),
            strategy='best1bin',
            maxiter=30,
            popsize=15,
            recombination=0.7,
            disp=True
        )
        
        if result.success:
            print(f"Optimization succeeded | Loss: {result.fun:.4f}")
            print(f"α: {result.x[0]:.2e}, β: {result.x[1]:.3f}")
            return result.x
        else:
            print("Optimization warning:", result.message)
            return [5e-5, 1.0]  # Fallback to reasonable defaults
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        return [5e-5, 1.0]

def get_optimized_physics_predictions(X_train, Y_train, X_val, X_test, 
                                     norm_stats, col_map, H):
    """Get physics-based predictions with optimized parameters"""
    # Step 1: Optimize parameters
    opt_params = optimize_physics_model(X_train, Y_train, norm_stats, col_map, H)
    alpha_opt, beta_opt = opt_params
    
    # Step 2: Generate predictions
    Y_train_phy = parameterized_physics_model(X_train, norm_stats, col_map, H, alpha_opt, beta_opt)
    Y_val_phy = parameterized_physics_model(X_val, norm_stats, col_map, H, alpha_opt, beta_opt)
    Y_test_phy = parameterized_physics_model(X_test, norm_stats, col_map, H, alpha_opt, beta_opt)
    
    return Y_train_phy, Y_val_phy, Y_test_phy, opt_params

# ====================
# USAGE EXAMPLE BELOW
# ====================

def evaluate_predictions(name, y_true, y_pred):
    """Helper function for evaluation"""
    flat_true = y_true.flatten()
    flat_pred = y_pred.flatten()
    mask = np.isfinite(flat_true) & np.isfinite(flat_pred)
    
    if mask.sum() == 0:
        print(f"{name}: No valid predictions")
        return 0, 0, 0
    
    flat_true = flat_true[mask]
    flat_pred = flat_pred[mask]
    
    mae = mean_absolute_error(flat_true, flat_pred)
    rmse = np.sqrt(mean_squared_error(flat_true, flat_pred))
    r2 = r2_score(flat_true, flat_pred)
    
    print(f"{name} Metrics:")
    print(f"- MAE: {mae:.3f}°C")
    print(f"- RMSE: {rmse:.3f}°C")
    print(f"- R²: {r2:.3f}\n")
    return mae, rmse, r2

# Example usage:
if __name__ == "__main__":
    # Mock data preparation (replace with real data)
    n_samples = 1000
    timesteps = 24
    n_features = 11  # 3 center + 8 neighbors
    H = 6  # 6-hour forecast
    
    # Mock datasets
    X_train = np.random.randn(n_samples, timesteps, n_features)
    Y_train = np.random.randn(n_samples, H)
    X_val = np.random.randn(200, timesteps, n_features)
    Y_val = np.random.randn(200, H)
    X_test = np.random.randn(300, timesteps, n_features)
    Y_test = np.random.randn(300, H)
    
    # Mock normalization stats
    norm_stats = {
        'mean': np.zeros(n_features),
        'std': np.ones(n_features)
    }
    
    # Column mapping - MUST match your feature order!
    col_map = {
        0: 'C_T2M',
        1: 'C_WS50M',
        2: 'C_WD50M',
        3: 'N_T2M',
        4: 'NE_T2M',
        5: 'E_T2M',
        6: 'SE_T2M',
        7: 'S_T2M',
        8: 'SW_T2M',
        9: 'W_T2M',
        10: 'NW_T2M'
    }
    
    # Get physics-based predictions
    print("Starting physics model optimization...")
    Y_train_phy, Y_val_phy, Y_test_phy, opt_params = get_optimized_physics_predictions(
        X_train, Y_train, X_val, X_test, norm_stats, col_map, H
    )
    
    # Evaluate results
    print("\nEvaluation Results:")
    train_metrics = evaluate_predictions("Train", Y_train, Y_train_phy)
    val_metrics = evaluate_predictions("Validation", Y_val, Y_val_phy)
    test_metrics = evaluate_predictions("Test", Y_test, Y_test_phy)
    
    # Show optimized parameters
    alpha, beta = opt_params
    print(f"Optimized Parameters:")
    print(f"- Thermal diffusivity (α): {alpha:.2e} m²/s")
    print(f"- Update scaling (β): {beta:.3f}")