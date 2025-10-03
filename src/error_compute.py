import numpy as np

def calculate_combined_errors(Y, Y_hat):
    """
    Calculate combined RMSE+MAE error for each example.
    
    Args:
        Y (ndarray): True values of shape (examples, H)
        Y_hat (ndarray): Predicted values of shape (examples, H)
    
    Returns:
        tuple: (number of examples, error vector where each element is RMSE+MAE for that example)
    """
    # Calculate RMSE per example (across H dimension)
    rmse_per_example = np.sqrt(np.mean((Y - Y_hat)**2, axis=1))
    
    # Calculate MAE per example (across H dimension)
    mae_per_example = np.mean(np.abs(Y - Y_hat), axis=1)
    
    # Combine errors (RMSE + MAE) for each example
    combined_errors = (rmse_per_example + mae_per_example)/2
    
    # Return number of examples and error vector
    return  combined_errors