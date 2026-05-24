# TAHMO Incoming Solar Radiation Prediction

**Author:** [Venon Takunda Nyadombo](https://github.com/venontn)

Predict **incoming shortwave radiation** (`radiation (W/m2)`) at **15-minute** resolution for **even months** of year 1 at each station, using odd-month radiation (training) and full-year weather variables.

Competition: [Zindi — TAHMO Solar Radiation](https://zindi.africa/competitions/tahmo-incoming-solar-radiation-prediction-challenge)

Repository: https://github.com/venontn/tahmo-solar-radiation

## Google Drive + Colab

1. Install [Google Drive for Desktop](https://www.google.com/drive/download/) (optional), or upload the project folder manually to **My Drive → `tahmo-solar-radiation`** (include `data/*.csv`).
2. Sync from Windows (if Drive is installed):

```powershell
.\scripts\sync_to_google_drive.ps1
```

3. Open [`notebooks/tahmo_colab_drive.ipynb`](notebooks/tahmo_colab_drive.ipynb) in [Google Colab](https://colab.research.google.com/), mount Drive, and run all cells.

## Setup (local)

1. Download from Zindi into `data/`:
   - `Train.csv`
   - `Test.csv`
   - `SampleSubmission.csv`
   - (optional) `dataset_data_dictionary.csv`, `Reference.csv`

2. Install dependencies:

```powershell
cd C:\Users\PC\Projects\tahmo-solar-radiation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
# Quick validation (leave-one-odd-month-out)
python src/train_predict.py --cv

# Build submission
python src/train_predict.py --blend 0.55
```

Output: `output/submission.csv` with columns `ID`, `TargetMBE`, `TargetRMSE` (same values per row, as required).

## Approach

| Component | Role |
|-----------|------|
| **Per-station LightGBM** | Maps weather + solar geometry → radiation (generalizes across months) |
| **Analog climatology** | For each even month, averages radiation from **adjacent odd months** at the same (day, hour, minute) |
| **Blend** | Default 55% ML / 45% analog (tune with `--blend`) |
| **Post-processing** | Clip to `[0, 1400]` W/m²; force 0 when sun below horizon |

### Metrics (Zindi)

- **\|MBE\|** = \|mean(pred − true)\| — control systematic bias (50% of score)
- **RMSE** — control magnitude of errors (50% of score)

Tune blend and model hyperparameters to balance both.

## Next improvements

- Add **satellite** shortwave (e.g. ERA5, CAMS, Meteosat) — required/encouraged by organizers
- Rolling/lag features on temperature, humidity, precipitation
- Station-level bias calibration on held-out odd months
- Ensemble multiple seeds / models

## Data layout

Training rows include **odd months only** with `radiation (W/m2)`. Test rows are **even months** without radiation; weather columns are present for the full year.
