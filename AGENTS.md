# Repository Guidelines

## Project Structure & Module Organization

This repository is a compact ML anomaly-detection project. Source scripts are in `code/`, with versioned training/prediction pipelines such as `train_predict_v3_fast.py` and `train_predict_v4_full.py`. Input datasets live in `data/`: `train.csv`, `test_simple.csv`, and `test_complex.csv`. Generated prediction artifacts and serialized models are stored in `submission/`, `submission_v3/`, and any new `submission_v*` directory. `Project.pdf` is the task reference, and `.gitattributes` tracks large datasets and model files through Git LFS.

## Build, Test, and Development Commands

Run commands from the repository root so relative paths like `data/train.csv` resolve correctly.

```powershell
python code\train_predict_v3_fast.py
```

Trains the v3 fast pipeline and writes `submission_v3/pred_simple.csv`, `submission_v3/pred_complex.csv`, and `submission_v3/model.pkl`.

```powershell
python code\train_predict_v4_full.py
```

Runs the full v4 pipeline and writes outputs to `submission_v4/`.

If dependencies are missing, install the observed runtime stack in your active environment:

```powershell
pip install pandas numpy scikit-learn xgboost lightgbm
```

## Coding Style & Naming Conventions

Use Python with 4-space indentation and concise, descriptive function names. Keep pipeline constants near the top of each script, following the existing pattern: `TRAIN_PATH`, `TEST_SIMPLE_PATH`, `TEST_COMPLEX_PATH`, and `OUTPUT_DIR`. Use snake_case for functions and variables, and reserve uppercase for configuration constants. Prefer vectorized pandas/NumPy operations over row-by-row loops when adding feature engineering.

## Testing Guidelines

There is no dedicated test suite in this checkout. Validate changes by running the affected training script end to end and confirming that prediction CSVs are created with a single `y_pred` column and the expected row counts. For faster iteration, modify or add a smaller experimental script rather than overwriting a known working version.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, for example `Upload v4 full implementation with all optimizations` and `Add v4 optimization code placeholder`. Keep commit messages direct and scoped to one change. Pull requests should describe the pipeline version changed, list regenerated artifacts, report validation metrics or runtime observations, and note whether large files require Git LFS.

## Security & Configuration Tips

Do not hardcode credentials or local absolute paths in training scripts. Keep datasets, PDFs, pickles, and other large binary artifacts under Git LFS, and avoid committing regenerated `submission_v*` outputs unless they are part of the intended deliverable.
