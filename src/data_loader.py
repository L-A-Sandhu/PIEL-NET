import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(csv_path, T, S, H, output_dir=None):
    """
    Load and process time series data for forecasting
    
    Args:
        csv_path: Path to CSV file
        T: Input sequence length (number of time steps)
        S: Step size between sequences
        H: Forecast horizon
        output_dir: Directory to save output (default: CSV filename without extension)
    
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, normalization_stats, column_mapping
    """
    # Step 1: Read CSV and create output directory
    df = pd.read_csv(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = output_dir or base_name
    os.makedirs(output_dir, exist_ok=True)
    
    # Create column mapping
    column_mapping = {i: col for i, col in enumerate(df.columns)}
    
    # Step 2: Split data (70% train, 15% validation, 15% test)
    n = len(df)
    train_end = int(0.7 * n)
    val_end = train_end + int(0.15 * n)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # Step 3: Normalize data using Z-score
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_df)
    val_norm = scaler.transform(val_df)
    test_norm = scaler.transform(test_df)
    
    # Save normalization parameters
    normalization_stats = {
        'mean': scaler.mean_.tolist(),
        'std': scaler.scale_.tolist(),
        'columns': df.columns.tolist()
    }
    with open(os.path.join(output_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(normalization_stats, f, indent=4)
    
    # Step 4: Find target column index
    target_col = 'C_T2M'
    target_idx = df.columns.get_loc(target_col)
    
    # Helper function to create sequences
    def create_sequences(data, T, S, H, target_idx):
        X, Y = [], []
        n = len(data)
        start = 0
        while start + T + H <= n:
            X.append(data[start:start+T])
            Y.append(data[start+T:start+T+H, target_idx])
            start += S
        return np.array(X), np.array(Y)
    
    # Step 5-6: Create sequences for each dataset
    X_train, Y_train = create_sequences(train_norm, T, S, H, target_idx)
    X_val, Y_val = create_sequences(val_norm, T, S, H, target_idx)
    X_test, Y_test = create_sequences(test_norm, T, S, H, target_idx)
    
    # Print dataset statistics
    def print_stats(name, X, Y):
        print(f"\n{name} Statistics:")
        print(f"X shape: {X.shape}")
        print(f"X min: {X.min():.4f}, X max: {X.max():.4f}, X mean: {X.mean():.4f}")
        print(f"Y shape: {Y.shape}")
        print(f"Y min: {Y.min():.4f}, Y max: {Y.max():.4f}, Y mean: {Y.mean():.4f}")
        
    print_stats("Training", X_train, Y_train)
    print_stats("Validation", X_val, Y_val)
    print_stats("Test", X_test, Y_test)
    
    # Print column mapping
    print("\nColumn Mapping:")
    for idx, col in column_mapping.items():
        print(f"{idx:2d}: {col}")
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, normalization_stats, column_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time Series Data Loader')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--T', type=int, required=True, help='Input sequence length')
    parser.add_argument('--S', type=int, required=True, help='Step size between sequences')
    parser.add_argument('--H', type=int, required=True, help='Forecast horizon')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Load and process data
    results = load_data(
        csv_path=args.csv_path,
        T=args.T,
        S=args.S,
        H=args.H,
        output_dir=args.output_dir
    )
    
    # Unpack results
    X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_stats, col_map = results
    
    # Save column mapping
    if args.output_dir:
        with open(os.path.join(args.output_dir, 'column_mapping.json'), 'w') as f:
            json.dump(col_map, f, indent=4)
        print(f"\nSaved column mapping to directory: {args.output_dir}")

# Example usage when imported
# from data_loader import load_data
# X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_stats, col_map = load_data(...)