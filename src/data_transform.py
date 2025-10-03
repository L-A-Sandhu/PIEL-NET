import numpy as np

def transform_X(X, col_map):
    """
    Transform input sequences by adding repeated statistics channel after HR feature.
    
    Args:
        X: Input array of shape (num_examples, T, num_features)
        col_map: Dictionary mapping column indices to feature names
        
    Returns:
        X_transformed: Array of shape (num_examples, T, 5, 10) with reorganized features
        new_col_map: Updated column mapping for the first group
    """
    # Find index of C_T2M feature
    target_idx = None
    for idx, name in col_map.items():
        if name == 'C_T2M':
            target_idx = idx
            break
    if target_idx is None:
        raise ValueError("C_T2M column not found in column mapping")

    num_examples, T, num_features = X.shape
    # Create new array with space for Stat channel (+1 feature)
    X_new = np.zeros((num_examples, T, num_features + 1))
    
    # Build new column mapping for first group
    new_col_map = {0: 'YEAR', 1: 'MO', 2: 'DY', 3: 'HR', 4: 'Stat'}
    
    # Process each example
    for i in range(num_examples):
        # Extract C_T2M values for this sequence
        c_t2m_vals = X[i, :, target_idx]
        
        # Compute statistics
        mean_val = np.mean(c_t2m_vals)
        var_val = np.var(c_t2m_vals)
        vmr_val = var_val / (mean_val + 1e-7)  # Avoid division by zero
        
        # Create repeated stat sequence (3 stats × 16 repeats = 48 values)
        stat_sequence = np.tile([mean_val, var_val, vmr_val], T // 3)
        
        # Reorganize features:
        # 1. Preserve first 4 columns (YEAR, MO, DY, HR)
        # 2. Insert new Stat sequence as 5th column
        # 3. Shift original columns 4+ to new positions 5+
        X_new[i, :, 0:4] = X[i, :, 0:4]        # Time features
        X_new[i, :, 4] = stat_sequence          # New Stat channel
        X_new[i, :, 5:] = X[i, :, 4:]           # Original features (shifted right)
    
    # Reshape into (Examples, Time, Features, Location)
    X_transformed = X_new.reshape(num_examples, T, 5, 10)
    
    return X_transformed, new_col_map