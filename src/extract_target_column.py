def extract_target_column(X, col_map, target_col='C_T2M'):
    """
    Extracts values of a specific column from input sequences.
    
    Args:
        X: Input sequences (3D numpy array of shape (samples, timesteps, features))
        col_map: Column mapping dictionary (index to column name)
        target_col: Name of target column to extract (default: 'C_T2M')
    
    Returns:
        2D numpy array of extracted values (samples, timesteps)
    """
    # Find the index of the target column
    target_idx = None
    for idx, col_name in col_map.items():
        if col_name == target_col:
            target_idx = idx
            break
    
    if target_idx is None:
        raise ValueError(f"Target column '{target_col}' not found in column mapping")
    
    # Extract values for the target column across all samples and timesteps
    return X[:, :, target_idx]