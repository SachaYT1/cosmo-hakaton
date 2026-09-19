# Progress — fire monitoring case (KosmoHackathon 2026)

Status as of 2026-09-18. All numbers are from the training data (`data/train`).

## Summary

| Module | Method | CV result | Status |
|---|---|---|---|
| AF (active fire) | Rule: day I4 > 330 K; night I4 > 317 K and I4 at least 3 K above the 15×15 background | F1 0.866 (fitted on all train, not CV) | done, kept as reference |
| AF (active fire) | LightGBM on contextual candidates + weak NOAA history | **F1 0.953463** (3 date folds, 2019–2023) | done, **used in inference** |
| BS (burn severity) | Rule: 3×3 median dNBR, thresholds per landcover class | **IoU_burn 0.434, mIoU_sev 0.462** (5-fold, grouped by chip = fire event) | done, **used in inference** |
| BS (burn severity) | U-Net (burn outline + severity head) | — | code written, **not trained** |

A **valid `submission.csv`** for the test set is produced in **~8.5 s**:

```bash
python inference.py --data-dir data/test --output outputs/submission_rules.csv --bs-model rules
python -m scripts.validate_submission outputs/submission_rules.csv     # -> VALID
```

`--bs-model rules` is required for now. The default (`unet`) needs `configs/bs_unet.json` and trained weights, and neither exists yet.

The metric's combined Score can't be stated honestly yet: the AF and BS numbers come from different CV setups. Plugging them into the formula as a rough guide gives
`0.35·0.953 + 0.35·0.434 + 0.30·0.462 ≈ 0.62`.

## Step by step

### 1. Baseline and metric script
- **Not found** in `data/` or in the original `.tar` archives, even though the statement says they are issued with the data. Ask the organisers for them.
- Written from the spec instead:
  - `firemon/rle.py` — RLE encode/decode (1-based, row-major, runs must not overlap or touch). Round-trip tested on random masks.
  - `firemon/metric.py` — Score, micro-averaged over all pixels; a zero denominator counts as 1. **Not checked against the official script.**

### 2. EDA (`scripts/eda_bs.py` → `reports/`)
Data layout (the empty `pre/`, `post/`, `sar_*`, `images/` folders are leftovers from a rename):

| File | Shape / type | Bands |
|---|---|---|
| `bs/sentinel2_{pre,post}` | 512×512×10 uint16 | B2 B3 B4 B5 B6 B7 B8A B11 B12 SCL |
| `bs/sentinel1_{pre,post}` | 512×512×2 int16 (dB×100) | VV VH |
| `bs/aux` | 512×512×3 int16 | dem, slope, landcover (**no aspect**, although the statement lists it) |
| `bs/masks` | 512×512 uint8 | 0–3 |
| `af/viirs` | 256×256×8 float32 | I1–I5, solar_zenith, sensor_zenith, valid (I1–I3 are NaN at night) |
| `af/aux` | 256×256×5 float32 | landcover, dem, t2m, rh2m, wind_speed |

Findings and what we decided because of them:
1. **Severity inside a burn is almost a pure threshold** on 3×3-median dNBR, with cut-offs that depend on landcover. With the true burn mask it classifies **97.7 %** of burned pixels correctly (crop 98.5 %, grass 97.5 %, tree 91.9 %). The fitted cut-offs:

   | Landcover | 1→2 | 2→3 |
   |---|---|---|
   | tree | 0.27 | 0.57 |
   | grass | 0.20 | 0.39 |
   | crop | 0.18 | 0.38 |
   | wetland | 0.33 | 0.61 |

   → Severity is assigned by this rule. A learned model is only needed for the burn outline.
2. Pixels whose SCL is **nodata (0), cloud shadow (3) or high cloud (9)** are never labelled burned → they are always predicted 0.
3. **The burn outline is the hard part.** 99.3 % of burned pixels have median dNBR > 0.06, but a plain threshold only reaches IoU ≈ 0.42. About a third of the area in high-dNBR connected regions is completely unburned, and much of it lies more than 2 km from any fire (harvested or ploughed fields) → a spatial model is needed.
4. Every BS chip is its own fire event (224 chips, 224 events) → CV folds split by chip.
5. **The test set is cloudier than train.** The test 75th-percentile cloud fraction is 0.26, while in train 90 % of chips are below 0.25.
6. AF: 9,725 fire pixels (0.035 % of pixels). **Day fires** separate at I4 ≈ 330 K. **Night fires** are much cooler, so a fixed threshold catches only 15–30 % of them and a context test is needed. False positives almost never repeat at the same location in train.

Outputs: `reports/figures/bs_examples.png`, `reports/figures/bs_dnbr_by_landcover.png`, `reports/bs_eda.json`, `reports/bs_per_chip.csv`.

### 3. BS rule baseline (`scripts/fit_bs_rules.py`)
- Fits the severity cut-offs per landcover class, one global burn threshold (**0.12**) and a minimum component size (**200 px**). Classes with fewer than 20k burned pixels fall back to a default.
- 5-fold CV: IoU_burn 0.434, mIoU_sev 0.462 (sev1 0.228, sev2 0.479, sev3 0.679). Class 1 is weak because false positives from fields fall in the low-dNBR range.
- Output: `configs/bs_thresholds.json`, `configs/bs_thresholds_post.json`.

### 4. AF module (`firemon/af.py`, `scripts/train_af.py`)
- Features: I1–I5, I4−I5, zenith angles, day flag, the pixel's excess over its local mean and z-score in 7/15/31 px windows (for I4 and I4−I5), local-maximum features, landcover, DEM, weather, NDVI.
- Candidate filter (I4 > 320 K, or warmer than its surroundings) keeps 99.98 % of fire pixels and about 6.5 % of all pixels.
- LightGBM with seed 42. The decision threshold (0.44) is chosen on 2019–2023 out-of-fold predictions. **CV F1 0.953463.** Historical NOAA weak labels enter training with weight 0.02 and never enter validation. Most important features: I4 and I4−I5 excess over the local background.
- Fixed: fold groups used Python's `hash()`, which is randomised per process, so each run had different folds (thresholds 0.36 vs 0.20). Groups are now the date string.
- **Day vs night** (`scripts/eval_af.py` → `reports/af_daynight.json`). Night = chip median solar zenith ≥ 85°. Train is 30.2 % night chips, test 38.3 %.

  | | all | day | night | reweighted to test mix |
  |---|---|---|---|---|
  | Rule baseline | 0.867 | 0.897 | **0.534** (recall 0.37) | 0.855 |
  | LightGBM + NOAA weak labels (OOF) | 0.953 | — | — | — |

  Bootstrap 95 % intervals: day [0.946, 0.959], night [0.945, 0.972]. Separate day/night thresholds bring no gain: both come out at 0.43, and CV without leakage gives 0.952. F1 stays within 0.002 of the best for thresholds 0.25–0.50, so the operating point is stable. By satellite: SNPP 0.950, NOAA-20 0.958, NOAA-21 0.957 (only 17 chips).
- Active output: `weights/af_augmented_best_lgb.txt`, `configs/af.json`, `reports/af_augmented_best.json`. Losing AF runtime weights and configs were removed after the final LightGBM/CatBoost comparison; historical metrics remain in reports.

### 5. Inference
- `inference.py` → `firemon/pipeline.py`: parallel over chips. A chip that fails gets an empty mask instead of crashing the run. Rows follow `sample_submission.csv`, and `rle` is always quoted.
- `scripts/validate_submission.py` checks every rule in the statement: 447 rows, the set of (chip_id, class_id) pairs, RLE bounds and order, no overlap between classes.

### 6. Learned BS model — started, not finished
- `firemon/bs_unet.py`: 42-channel input (S2 pre/post, NBR/dNBR/dNDVI, S1 with differences, DEM/slope, landcover one-hot, SCL flags). The U-Net has 4 outputs: burn plus 3 severity classes. At inference the severity comes from either the net or the dNBR rule, whichever wins in CV. Base 16 = 1.9 M parameters, 0.3 s per chip on CPU.
- `scripts/prepare_bs.py` has run: `cache/bs_x.npy` (4.9 GB float16).
- `scripts/train_bs_unet.py`: 5-fold with the same seed-42 split, saves out-of-fold predictions to `cache/oof/`. **The first smoke run was killed (exit 137, probably out of memory with `--workers 3`). It has not been retried; try `--workers 0` next.**

## Not done
- U-Net training, tuning the burn threshold on out-of-fold predictions, `configs/bs_unet.json`.
- Web service (deliberately untouched).
- Dockerfile and slides.
- Checking our metric against the official script.

## Caveats
- Environment: Anaconda Python 3.12, numpy 1.26.4, xgboost 2.1.1, torch 2.6.0, opencv 4.10, tifffile 2023.4.12, pandas 2.2.2. Not pinned in a file yet.
- Disk: about 3.8 GB free. `data/*.tar` (3.2 GB) duplicates the extracted data.
- A second Claude session edited some of these files at the same time (`firemon/af.py`, `inference.py`, `scripts/train_bs_unet.py`) and retrained AF. The current files are consistent, and the numbers above match the current weights.

## File map
```
inference.py                 entry point → submission.csv
firemon/  io.py rle.py metric.py features.py cache.py bs_rules.py af.py pipeline.py bs_unet.py
scripts/  eda_bs.py fit_bs_rules.py train_af.py prepare_bs.py train_bs_unet.py validate_submission.py
configs/  af.json bs_thresholds.json bs_thresholds_post.json
weights/  af_augmented_best_lgb.txt
reports/  EDA figures and stats        outputs/  submission_rules.csv        cache/  training caches
```
