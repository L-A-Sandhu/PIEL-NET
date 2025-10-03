

import os
import argparse
import numpy as np
import json
import sys

import sys
sys.path.insert(0, './src')
from data_loader import load_data                                
from advection import get_optimized_physics_predictions          
from matric import calculate_metrics                              
from data_transform import transform_X                             
from error_compute import calculate_combined_errors                
from extract_target_column import extract_target_column           
from fuzzy_mem import compute_fuzzy_memberships                    
from PIEL_NET import HybridModel                                   
from ed_rvfl_sc import edRVFL_SC

def parse_args():
    p = argparse.ArgumentParser(description="PIELNET pipeline (save only final ensemble)")
    # Data + IO
    p.add_argument("--data_dir", type=str, default="./Data",
                   help="Directory with input CSV files (processed in batch).")
    p.add_argument("--results_root", type=str, default="Results/Ablation",
                   help="Root directory to save outputs.")
    p.add_argument("--output_name", type=str, default="C_T2M",
                   help="Target column name for metrics and denorm.")
    # Sequence config (defaults preserved)
    p.add_argument("--T", type=int, default=48, help="Input window length.")
    p.add_argument("--S", type=int, default=12, help="Stride between windows.")
    p.add_argument("--H", type=int, default=12, help="Forecast horizon.")
    # Training config (kept at previous defaults)
    p.add_argument("--epochs", type=int, default=1000, help="Max epochs for TF models.")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    # HybridModel loss + weights (defaults match class)
    p.add_argument("--loss_type_v4", type=str, default="MAE", choices=["MAE", "MSE", "focal"],
                   help="Loss for V4 data model.")
    p.add_argument("--loss_type_v5", type=str, default="focal", choices=["MAE", "MSE", "focal"],
                   help="Loss for V5 data model.")
    p.add_argument("--alpha_w", type=float, default=0.2, help="Weight α in custom loss.")
    p.add_argument("--beta_w", type=float, default=2.0, help="Weight β in custom loss.")
    p.add_argument("--gamma_w", type=float, default=5.0, help="Weight γ in custom loss.")
    p.add_argument("--eta", type=float, default=0.1, help="Stability term η in custom loss.")
    p.add_argument("--focal_gamma", type=float, default=5.0, help="Focal γ for confidence shaping.")
    # MoE / ED-RVFL-SC defaults (as before in your script)
    p.add_argument("--rvfl_units", type=int, default=256)
    p.add_argument("--rvfl_lambda", type=float, default=1e-3)
    p.add_argument("--rvfl_Lmax", type=int, default=7)
    p.add_argument("--rvfl_deep_boost", type=float, default=0.9)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def main():
    args = parse_args()
    np.random.seed(args.seed)

    ensure_dir(args.results_root)

    for csv_file in sorted(os.listdir(args.data_dir)):
        if not csv_file.endswith(".csv"):
            continue

        csv_path = os.path.join(args.data_dir, csv_file)
        dataset = os.path.splitext(csv_file)[0]  # dataset name from file
        run_dir = ensure_dir(os.path.join(args.results_root, dataset))
        final_dir = ensure_dir(os.path.join(run_dir, "PIELNET"))  # unified folder name

        print(f"\n=== Processing dataset: {dataset} ===")

        # ---- Load & window ----
        X_train, Y_train, X_val, Y_val, X_test, Y_test, norm_stats, col_map = load_data(
            csv_path=csv_path, T=args.T, S=args.S, H=args.H, output_dir=None
        )

        # ---- Physics predictions ----
        Y_train_phy, Y_val_phy, Y_test_phy, _ = get_optimized_physics_predictions(
            X_train, Y_train, X_val, X_test, norm_stats, col_map, args.H
        )

        # Align physics predictions via linear map (as before)
        A = np.linalg.pinv(Y_train_phy) @ Y_train
        Y_train_phy = Y_train_phy @ A
        Y_val_phy   = Y_val_phy   @ A
        Y_test_phy  = Y_test_phy  @ A

        # ---- LMS baseline over raw features ----
        Xtr = X_train.reshape(X_train.shape[0], -1)
        Xva = X_val.reshape(X_val.shape[0], -1)
        Xte = X_test.reshape(X_test.shape[0], -1)
        W_lms, *_ = np.linalg.lstsq(Xtr, Y_train, rcond=None)
        Y_est_train = Xtr @ W_lms
        Y_est_val   = Xva @ W_lms
        Y_est_test  = Xte @ W_lms

        # ---- Physics + LMS hybrid projection ----
        Y_train_H = np.hstack([Y_est_train, Y_train_phy])
        Y_val_H   = np.hstack([Y_est_val,   Y_val_phy])
        Y_test_H  = np.hstack([Y_est_test,  Y_test_phy])

        A2 = np.linalg.pinv(Y_train_H) @ Y_train
        Y_train_H = Y_train_H @ A2
        Y_val_H   = Y_val_H   @ A2
        Y_test_H  = Y_test_H  @ A2

        # ---- Error signals & memberships (no saving) ----
        Err_train = calculate_combined_errors(Y_train_H, Y_train)    # :contentReference[oaicite:18]{index=18}
        Err_val   = calculate_combined_errors(Y_val_H,   Y_val)
        # Not used later but available if needed:
        _ = extract_target_column(X_train, col_map, target_col=args.output_name)  # :contentReference[oaicite:19]{index=19}
        train_memberships, val_memberships = compute_fuzzy_memberships(
            Err_train, Err_val, mf_type="triangle"
        )  # :contentReference[oaicite:20]{index=20}

        # ---- Data transform for ConvLSTM model ----
        X_train_P = np.expand_dims(transform_X(X_train, col_map)[0], axis=-1)  # :contentReference[oaicite:21]{index=21}
        X_val_P   = np.expand_dims(transform_X(X_val,   col_map)[0], axis=-1)
        X_test_P  = np.expand_dims(transform_X(X_test,  col_map)[0], axis=-1)

        # Pack labels with error + memberships (as before)
        YY_train = np.column_stack((Y_train, Err_train, train_memberships))
        YY_val   = np.column_stack((Y_val,   Err_val,   val_memberships))

        # ---- Data model V4 (MAE by default) — no checkpoints saved ----
        data_model = HybridModel(
            input_shape=X_train_P.shape[1:],
            pi_dim=Y_train.shape[1],
            checkpoint_path=os.path.join(run_dir, "_tmp_ignore_v4"),
            loss_type=args.loss_type_v4,
            alpha=args.alpha_w, beta=args.beta_w, gamma=args.gamma_w,
            eta=args.eta, focal_gamma=args.focal_gamma
        )  # :contentReference[oaicite:22]{index=22}
        data_model.fit(
            X_train_P, YY_train,
            validation_data=(X_val_P, YY_val),
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
        )
        Y_train_D = data_model.predict(X_train_P)
        Y_val_D   = data_model.predict(X_val_P)
        Y_test_D  = data_model.predict(X_test_P)

        # ---- Data model V5 (focal by default) — no checkpoints saved ----
        data_model_F = HybridModel(
            input_shape=X_train_P.shape[1:],
            pi_dim=Y_train.shape[1],
            checkpoint_path=os.path.join(run_dir, "_tmp_ignore_v5"),
            loss_type=args.loss_type_v5,
            alpha=args.alpha_w, beta=args.beta_w, gamma=args.gamma_w,
            eta=args.eta, focal_gamma=args.focal_gamma
        )
        data_model_F.fit(
            X_train_P, YY_train,
            validation_data=(X_val_P, YY_val),
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
        )
        Y_train_FD = data_model_F.predict(X_train_P)
        Y_val_FD   = data_model_F.predict(X_val_P)
        Y_test_FD  = data_model_F.predict(X_test_P)

        # ---- Final ensemble (formerly V6) ----
        Z_train = np.hstack([Y_train_D, Y_train_FD, Y_train_H])
        Z_val   = np.hstack([Y_val_D,   Y_val_FD,   Y_val_H])
        Z_test  = np.hstack([Y_test_D,  Y_test_FD,  Y_test_H])

        MOE = edRVFL_SC(
            num_units=args.rvfl_units,
            activation="relu",
            lambda_=args.rvfl_lambda,
            Lmax=args.rvfl_Lmax,
            deep_boosting=args.rvfl_deep_boost
        )
        MOE.train(Z_train, Y_train)
        Y_test_final = MOE.predict(Z_test)

        # ---- Persist only final metrics under PIELNET ----
        metrics_all = {}
        metrics_all["PIELNET"] = calculate_metrics(
            Y_test, Y_test_final, args.output_name, norm_stats, final_dir
        )  # persists metrics.json and preds/true for the final only

        # Also save a compact metrics.json at dataset root for quick scan
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics_all, f, indent=4)

        print(f"✓ Completed {dataset} → {final_dir}")

if __name__ == "__main__":
    main()
