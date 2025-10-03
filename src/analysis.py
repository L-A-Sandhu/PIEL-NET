import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

def analyze_model_performance(X, Y, Y_data, Y_phy, col_map, output_path=None):
    """
    Comprehensive model performance analysis with horizon comparison and density visualization
    
    Args:
        X: Input features (num_samples, T, num_features)
        Y: True target values (num_samples, H)
        Y_data: Data-driven model predictions (num_samples, H)
        Y_phy: Physics model predictions (num_samples, H)
        col_map: Column index to name mapping
        output_path: Path to save plot (optional)
    """
    # Find C_T2M column index
    col_to_idx = {col: idx for idx, col in col_map.items()}
    c_t2m_idx = col_to_idx['C_T2M']
    
    # Extract C_T2M sequences
    c_t2m_sequences = X[:, :, c_t2m_idx]
    means = np.mean(c_t2m_sequences, axis=1)
    variances = np.var(c_t2m_sequences, axis=1)
    
    # 1. Horizon-wise performance comparison
    H = Y.shape[1]
    rmse_data = []
    rmse_phy = []
    physics_better_count = np.zeros(H)
    
    # Define RMSE calculation function compatible with all sklearn versions
    def calculate_rmse(true, pred):
        return np.sqrt(np.mean((true - pred) ** 2))
    
    for h in range(H):
        # Calculate RMSE for this horizon
        rmse_data.append(calculate_rmse(Y[:, h], Y_data[:, h]))
        rmse_phy.append(calculate_rmse(Y[:, h], Y_phy[:, h]))
        
        # Count samples where physics model is better
        physics_better = np.abs(Y[:, h] - Y_phy[:, h]) < np.abs(Y[:, h] - Y_data[:, h])
        physics_better_count[h] = np.sum(physics_better)
    
    # 2. Overall performance comparison
    physics_better_fraction = physics_better_count / len(Y)
    
    # 3. Create density plots for conditions where each model excels
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot 1: Horizon-wise RMSE comparison
    ax1 = fig.add_subplot(gs[0, :])
    horizons = np.arange(1, H+1)
    width = 0.35
    
    ax1.bar(horizons - width/2, rmse_data, width, label='Data-Driven Model')
    ax1.bar(horizons + width/2, rmse_phy, width, label='Physics Model')
    
    # Add performance ratio text
    for i, h in enumerate(horizons):
        ratio = rmse_phy[i] / rmse_data[i]
        ax1.text(h, max(rmse_data[i], rmse_phy[i]) * 1.05, 
                f'{ratio:.2f}', ha='center', fontsize=9)
    
    ax1.set_xlabel('Prediction Horizon (hours ahead)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Horizon-wise Model Performance')
    ax1.set_xticks(horizons)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Physics model superiority frequency by horizon
    ax2 = fig.add_subplot(gs[1, 0])
    
    ax2.bar(horizons, physics_better_fraction * 100, color='green')
    ax2.axhline(50, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Prediction Horizon')
    ax2.set_ylabel('Physics Model Superiority (%)')
    ax2.set_title('Physics Model Superiority Frequency by Horizon')
    ax2.set_xticks(horizons)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Density plot of conditions
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate per-sample performance
    rmse_data_sample = np.sqrt(np.mean((Y - Y_data)**2, axis=1))
    rmse_phy_sample = np.sqrt(np.mean((Y - Y_phy)**2, axis=1))
    performance_ratio = rmse_phy_sample / rmse_data_sample
    
    # Create density plot
    xy = np.vstack([means, variances])
    z = gaussian_kde(xy)(xy)
    
    sc = ax3.scatter(means, variances, c=performance_ratio, 
                    cmap='coolwarm', alpha=0.6, 
                    vmin=0.5, vmax=1.5)
    
    ax3.set_xlabel('Mean Temperature (C_T2M)')
    ax3.set_ylabel('Temperature Variance (C_T2M)')
    ax3.set_title('Performance Ratio (Physics RMSE / Data RMSE)')
    fig.colorbar(sc, ax=ax3, label='RMSE Ratio')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance ratio distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(performance_ratio, bins=50, color='purple', alpha=0.7)
    ax4.axvline(1.0, color='red', linestyle='--')
    ax4.set_xlabel('Physics RMSE / Data RMSE')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Performance Ratio Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Characteristic comparison
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Identify samples where physics is significantly better/worse
    physics_better = performance_ratio < 0.95
    data_better = performance_ratio > 1.05
    
    characteristics = {
        'Mean Temp': means,
        'Temp Variance': variances,
        'Humidity': np.mean(X[:, :, col_to_idx['C_QV2M']], axis=1),
        'Wind Speed': np.mean(X[:, :, col_to_idx['C_WS50M']], axis=1)
    }
    
    physics_avg = [np.mean(characteristics[k][physics_better]) for k in characteristics]
    data_avg = [np.mean(characteristics[k][data_better]) for k in characteristics]
    
    x = np.arange(len(characteristics))
    width = 0.35
    
    ax5.bar(x - width/2, physics_avg, width, label='Physics Better')
    ax5.bar(x + width/2, data_avg, width, label='Data Better')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(characteristics.keys())
    ax5.set_ylabel('Average Value')
    ax5.set_title('Characteristic Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add overall statistics
    stats_text = (
        f"Physics Better: {np.mean(physics_better)*100:.1f}% of samples\n"
        f"Data Better: {np.mean(data_better)*100:.1f}% of samples\n"
        f"Overall Physics RMSE: {np.mean(rmse_phy):.4f}\n"
        f"Overall Data RMSE: {np.mean(rmse_data):.4f}"
    )
    plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save or show plot
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    return {
        'horizon_rmse_data': rmse_data,
        'horizon_rmse_phy': rmse_phy,
        'performance_ratio': performance_ratio,
        'physics_better_fraction': physics_better_fraction,
        'characteristics_comparison': {
            'physics_better': {k: v for k, v in zip(characteristics.keys(), physics_avg)},
            'data_better': {k: v for k, v in zip(characteristics.keys(), data_avg)}
        }
    }

# # Usage example
# results = analyze_model_performance(
#     X=X_test,  
#     Y=Y_test,   
#     Y_data=Y_test_data_model,  
#     Y_phy=Y_test_phy,          
#     col_map=col_map,
#     output_path='comprehensive_performance_analysis.png'
# )