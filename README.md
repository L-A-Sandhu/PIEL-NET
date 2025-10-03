# PIEL-NET: Physics‑Informed Ensemble Learning for City–Region Temperature Forecasting

*A production‑ready, research‑grade repository for physics‑informed machine learning (PIML), spatio‑temporal forecasting, and extreme‑event early warning at the city–region scale.*

**Keywords:** physics‑informed ML, ConvLSTM, ensemble learning, RAFL, advection–diffusion prior, urban climate analytics, heatwave/cold‑snap detection, NASA POWER, reproducible research, edge‑friendly inference.

## Overview
PIEL‑NET fuses a light advection–diffusion **physics prior** with a spatio‑temporal ConvLSTM **baseline** and a RAFL‑specialized **expert**, then blends them via an **edRVFL‑SC** ensemble for robust 12‑h temperature forecasts under normal and extreme regimes.

## Repository layout
```
advection.py              # advection–diffusion prior
analysis.py               # horizon/density diagnostics & plots
data_loader.py            # IO, windowing (T), stride (S), horizon (H), splits, scaling
data_transform.py         # tensor reshapes for Conv3D/ConvLSTM
error_compute.py          # residual & composite error utilities
extract_target_column.py  # helpers for target extraction
fuzzy_mem.py              # fuzzy memberships for RAFL
matric.py                 # metrics (MAE, RMSE, R², MAPE, WAPE, SMAPE)
PIEL_NET.py               # ConvLSTM baseline + RAFL‑specialized expert + edRVFL‑SC
main.py                   # end‑to‑end runner (iterates CSVs, saves final results only)
requirements.txt
```
> Results are written to `Results/<DATASET>/PIELNET/` as `metrics.json` and predictions for the final fused model.

## Data schema (hourly rows)
Leading timestamps: `YEAR, MO, DY, HR`  
For each site in `{C, N, NE, E, SE, S, SW, W, NW}` provide five features:  
`<SITE>_WS50M, <SITE>_WD50M, <SITE>_PS, <SITE>_QV2M, <SITE>_T2M`  
Example columns include: `C_WS50M, C_WD50M, …, C_T2M, N_WS50M, …, NW_T2M`.

- **Default target:** `C_T2M` (2‑m air temperature at city center).  
- **Default windowing:** lookback **T=48**, forecast **H=12** (hours).

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Quick start
Place your CSV files in `./Data/`. Each filename defines the dataset name.

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

## Run on your own dataset
1. **Match the schema** above (timestamps + 9 sites × 5 vars).  
2. **Save** as `./Data/MyCity.csv`.  
3. **Run** one of the commands; outputs appear in `Results/MyCity/PIELNET/`.

**Notes**
- Change the prediction target via `--output_name` if needed (e.g., `E_T2M`).  
- `data_loader.py` handles normalization and non‑overlapping splits to avoid leakage.

## Results & analysis
Use `analysis.py` to generate horizon‑wise plots, density‑conditioned error curves, and extreme‑zone diagnostics for model comparison and reporting.

## Citation
If you use this code, please cite the accompanying manuscript:  
**“Physics‑Informed Ensemble Learning for City‑Region Temperature Prediction During Thermal Extremes.”**

## License
This repository is released for research and educational use. See LICENSE (add one if needed).
