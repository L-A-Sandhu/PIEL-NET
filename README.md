# PIEL-NET: Physics‑Informed Ensemble Learning for City–Region Temperature Forecasting

[![Status](https://img.shields.io/badge/status-research--grade-1064ff)](#)
[![Paper](https://img.shields.io/badge/paper-PIEL--NET-informational)](#)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-2.x-red)
![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)
![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)
![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20macOS%20%7C%20Win64-black)

> **Short description:** Physics‑informed ensemble for 12‑hour city–region temperature forecasts. Advection–diffusion prior + ConvLSTM + RAFL + edRVFL‑SC for extreme‑event early warnings.

**Keywords:** physics‑informed ML, ConvLSTM, ensemble learning, RAFL, advection–diffusion prior, urban climate analytics, heatwave/cold‑snap detection, NASA POWER, reproducible research, edge‑friendly inference, spatio‑temporal forecasting, early‑warning system, city‑region climate, extreme‑event modeling.

---

## Why this repo
PIEL‑NET fuses a light **advection–diffusion physics prior** with a spatio‑temporal **ConvLSTM baseline** and a **RAFL‑specialized expert**, then blends them via an **edRVFL‑SC** ensemble. The pipeline is engineered for **robust tails** (heatwaves/cold snaps) while maintaining **low average error** and **portable inference**.

## Repository layout
```
advection.py              # learned advection–diffusion prior
analysis.py               # horizon/density diagnostics (optional)
data_loader.py            # CSV IO, windowing/splits, normalization
data_transform.py         # tensor reshapes for 3D Conv + ConvLSTM
error_compute.py          # residual & composite error utilities
extract_target_column.py  # helpers for target extraction
fuzzy_mem.py              # fuzzy memberships (triangular) for RAFL
matric.py                 # MAE, RMSE, R², MAPE, WAPE, SMAPE
PIEL_NET.py               # HybridModel: ConvLSTM baseline + RAFL expert + fusion
main.py                   # end‑to‑end runner (save final results only)
requirements.txt
```

**Outputs:** `Results/<DATASET>/PIELNET/metrics.json` plus `preds.npy`, `truth.npy` for the final fused model.

---

## Data schema (hourly rows)
Leading timestamp columns: `YEAR, MO, DY, HR`  
For each of 9 sites `{C, N, NE, E, SE, S, SW, W, NW}` supply five variables:
```
<SITE>_WS50M  <SITE>_WD50M  <SITE>_PS  <SITE>_QV2M  <SITE>_T2M
```
Example subset: `C_WS50M, C_WD50M, C_PS, C_QV2M, C_T2M, N_WS50M, …, NW_T2M`

- **Target (default):** `C_T2M`
- **Default sequence:** lookback **T=48**, stride **S=12**, horizon **H=12**

> The nine‑point layout (center + 8 neighbors ≈100 km) supplies advective context for the central grid cell.

---

## Installation
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
**requirements.txt**
```
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.8
scikit-learn>=1.3
tensorflow>=2.11
keras>=2.11
tqdm>=4.66
```
> Optional: If you maintain `ed_rvfl_sc` as a package, ensure it is importable (or add its module alongside).

---

## Quick start
1. Place one or more CSV datasets in `./Data/` (one city per file).
2. Run the pipeline; each CSV is processed in batch; only **final** results are persisted.

**Short command (sane defaults):**
```bash
python main.py --data_dir ./Data --results_root ./Results
```

**Long command (fully explicit & reproducible):**
```bash
python main.py \
  --data_dir ./Data --results_root ./Results/Ablation \
  --output_name C_T2M \
  --T 48 --S 12 --H 12 \
  --epochs 1000 --batch_size 32 --patience 10 \
  --loss_type_v4 MAE --loss_type_v5 focal \
  --alpha_w 0.2 --beta_w 2.0 --gamma_w 5.0 --eta 0.1 --focal_gamma 5.0 \
  --rvfl_units 256 --rvfl_lambda 1e-3 --rvfl_Lmax 7 --rvfl_deep_boost 0.9 \
  --seed 42
```

### CLI options
| Flag | Meaning | Default |
|---|---|---|
| `--data_dir` | Directory with CSV files | `./Data` |
| `--results_root` | Where to save outputs | `Results/Ablation` |
| `--output_name` | Target column for metrics/denorm | `C_T2M` |
| `--T` | Lookback window length (hours) | `48` |
| `--S` | Stride between windows (hours) | `12` |
| `--H` | Forecast horizon (hours) | `12` |
| `--epochs` | Max training epochs (TF parts) | `1000` |
| `--batch_size` | Batch size | `32` |
| `--patience` | Early‑stopping patience | `10` |
| `--loss_type_v4` | Loss for baseline data model | `MAE` |
| `--loss_type_v5` | Loss for RAFL expert | `focal` |
| `--alpha_w, --beta_w, --gamma_w` | Regime weights (low/med/high) | `0.2, 2.0, 5.0` |
| `--eta` | Stability term in RAFL | `0.1` |
| `--focal_gamma` | Focal exponent | `5.0` |
| `--rvfl_units` | Hidden width (edRVFL‑SC) | `256` |
| `--rvfl_lambda` | Ridge regularization | `1e-3` |
| `--rvfl_Lmax` | Depth (layers) | `7` |
| `--rvfl_deep_boost` | Skip/boost factor | `0.9` |
| `--seed` | Random seed | `42` |

---

## Run on custom data
1. Export hourly CSVs with the schema above (timestamps + 9 sites × 5 vars).  
2. Save as `./Data/MyCity.csv`.  
3. Launch the short or long command. Results go to `Results/MyCity/PIELNET/`.

**Tips**
- To predict a different site, set `--output_name E_T2M` (or any `<SITE>_T2M`).  
- Keep `T/S/H` consistent across datasets for fair transfer learning.

---

## Experiment artifacts
Each dataset produces:
```
Results/<DATASET>/PIELNET/
  metrics.json   # RMSE, MAE, R², MAPE, WAPE, SMAPE
  preds.npy      # predictions (denormalized)
  truth.npy      # ground truth (denormalized)
```
Use `analysis.py` to create horizon‑wise plots, density‑conditioned errors, and extreme‑zone diagnostics.

---

## Reproducibility
- Deterministic seeds via `--seed`.  
- Non‑overlapping splits in `data_loader.py` to avoid leakage.  
- Fixed schema and identical T/S/H across cities recommended for transfer tests.

---

## Troubleshooting
- **Missing `ed_rvfl_sc`**: install the package or copy its module into the repo root.  
- **CUDA/TF setup**: ensure TensorFlow **2.x** with a matching CUDA/cuDNN if using GPU.  
- **NaNs**: check CSV for missing values; impute or drop before training.

---

## How to cite
```
Aslam, L., Zou, R., Li, G., Awan, E. S., & Mouafik, S. (2025).
Physics‑Informed Ensemble Learning for City‑Region Temperature Prediction During Thermal Extremes.
```
BibTeX
```bibtex
@article{aslam2025pielnet,
  title   = {Physics-Informed Ensemble Learning for City-Region Temperature Prediction During Thermal Extremes},
  author  = {Aslam, Laeeq and Zou, Runmin and Li, Gang and Awan, E. S. and Mouafik, Sara},
  year    = {2025},
  journal = {Preprint}
}
```

## License
Apache‑2.0 (see LICENSE).
