# Repository Guidelines

## Project Structure & Module Organization

This repository is a time-series anomaly detection project with multiple versioned training pipelines. Core runnable scripts live in `code/`:

- `train_predict_v2.py`: early baseline, writes to `submission/`
- `train_predict_v3_fast.py`: stable v3 baseline, writes to `submission_v3/`
- `train_predict_v4_NoRegime.py`: current main v4 pipeline, writes to `submission_v4_NoRegime/`
- `train_predict_v4_Regime(ErrorVersion).py`: retained experimental/error-prone regime variant, do not treat as the default production path

Datasets are in `data/` with `train.csv`, `test_simple.csv`, and `test_complex.csv`. Analysis and handoff materials are split between `docs/` and `notebooks/`. Validation and reproducibility utilities live in `validation/`, including scripts for replaying saved models, notebook synchronization, and regime ablation experiments. Generated predictions and serialized models are kept in versioned `submission*` directories. `Project.pdf` is the task reference, and `.gitattributes` is used to track large data/model artifacts through Git LFS.

## Build, Test, and Development Commands

Run commands from the repository root so relative paths like `data/train.csv` resolve correctly.

```powershell
python code\train_predict_v3_fast.py
```

Runs the v3 baseline and writes `submission_v3/pred_simple.csv`, `submission_v3/pred_complex.csv`, and `submission_v3/model.pkl`.

```powershell
python code\train_predict_v4_NoRegime.py
```

Runs the current v4 no-regime pipeline and writes outputs to `submission_v4_NoRegime/`.

```powershell
python validation\validate_submission_v3_pkl.py
```

Replays the saved `submission_v3/model.pkl` against the validation split to check whether the saved artifact matches the documented v3 behavior.

```powershell
python validation\run_regime_ablation.py
```

Compares `global_regime`, `local_regime`, and `no_regime` v3 variants and refreshes `validation/regime_ablation_results.csv`.

```powershell
python validation\execute_notebook_locally.py
```

Executes the local v4 notebook in place without requiring `nbconvert`.

If dependencies are missing, install the observed project stack in your active environment:

```powershell
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```

If you need to regenerate notebooks or notebook-oriented analysis scripts, you may also need Jupyter-adjacent packages already used by the notebook files in this repo.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and snake_case for functions and variables. Keep configuration constants near the top of each script in uppercase, following the existing pattern such as `TRAIN_PATH`, `TEST_SIMPLE_PATH`, `TEST_COMPLEX_PATH`, `OUTPUT_DIR`, and version-specific split constants. Prefer concise pipeline helper functions over large inline blocks, and favor vectorized pandas/NumPy feature construction instead of row-wise loops. When adding a new pipeline version, create a new script or output directory instead of overwriting a known working baseline.

## Testing Guidelines

There is no dedicated unit-test suite in this checkout. Validate ML changes by running the affected pipeline end to end and confirming:

- prediction CSVs are created in the expected `submission*` directory
- each prediction file contains exactly one `y_pred` column
- row counts match the corresponding test set
- any saved `model.pkl` can still be consumed by the validation or notebook utilities that depend on it

For analysis-only changes, run the relevant helper in `validation/` instead of retraining every version. Prefer lightweight validation scripts for reproducibility checks before rerunning the most expensive full pipeline.

## Commit & Pull Request Guidelines

Use short imperative commit messages scoped to one change, consistent with the existing history. When a change affects a specific model version, mention that version explicitly in the commit summary. Pull requests should state which pipeline or validation workflow changed, list regenerated artifacts or notebooks, summarize validation metrics or runtime observations, and note any newly added large files that must stay under Git LFS.

## Security & Configuration Tips

Do not hardcode credentials, machine-specific absolute paths, or environment-specific secrets. Keep large CSVs, PDFs, pickles, and generated prediction artifacts under the existing LFS-aware workflow. Avoid committing regenerated `submission*` outputs unless they are part of the intended deliverable, and do not assume experimental files with `ErrorVersion` in the name are safe defaults without revalidation.
