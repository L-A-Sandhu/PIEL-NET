import numpy as np
import skfuzzy as fuzz

def compute_memberships(errors, mf_type, x_min, x_max, center_medium, spread, extended_min=None, extended_max=None):
    n = len(errors)
    memberships = np.zeros((n, 3))  # Columns: low, medium, high
    
    if mf_type == 'gaussian':
        memberships[:, 0] = fuzz.gaussmf(errors, x_min, spread)
        memberships[:, 1] = fuzz.gaussmf(errors, center_medium, spread)
        memberships[:, 2] = fuzz.gaussmf(errors, x_max, spread)
    elif mf_type == 'gbell':
        b = 4
        memberships[:, 0] = fuzz.gbellmf(errors, spread, b, x_min)
        memberships[:, 1] = fuzz.gbellmf(errors, spread, b, center_medium)
        memberships[:, 2] = fuzz.gbellmf(errors, spread, b, x_max)
    elif mf_type == 'triangle':
        if extended_min is None or extended_max is None:
            x_range = x_max - x_min
            extended_min = x_min - 0.1 * x_range
            extended_max = x_max + 0.1 * x_range
        memberships[:, 0] = fuzz.trimf(errors, [extended_min, x_min, center_medium])
        memberships[:, 1] = fuzz.trimf(errors, [x_min, center_medium, x_max])
        memberships[:, 2] = fuzz.trimf(errors, [center_medium, x_max, extended_max])
    return memberships

def compute_fuzzy_memberships(train_errors, validation_errors, mf_type):
    x_min = np.min(train_errors)
    x_max = np.max(train_errors)
    center_medium = (x_min + x_max) / 2.0
    spread = (x_max - x_min) / 4.0
    
    if mf_type == 'triangle':
        x_range = x_max - x_min
        extended_min = x_min - 0.1 * x_range
        extended_max = x_max + 0.1 * x_range
        train_memberships = compute_memberships(train_errors, mf_type, x_min, x_max, center_medium, spread, extended_min, extended_max)
        val_memberships = compute_memberships(validation_errors, mf_type, x_min, x_max, center_medium, spread, extended_min, extended_max)
    else:
        train_memberships = compute_memberships(train_errors, mf_type, x_min, x_max, center_medium, spread)
        val_memberships = compute_memberships(validation_errors, mf_type, x_min, x_max, center_medium, spread)
    
    return train_memberships, val_memberships

# Example usage:
# train_errors = np.random.rand(2553)  # Replace with actual data
# validation_errors = np.random.rand(2533)  # Replace with actual data
# mf_type = 'triangle'  # or 'gaussian' or 'gbell'
# train_memberships, val_memberships = compute_fuzzy_memberships(train_errors, validation_errors, mf_type)