# Electric Vehicle Purchase Prediction — Final Report

**Competition:** [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9)  
**Task:** Predict whether a customer will buy an electric vehicle  
**Metric:** ROC AUC (higher is better)  
**Status:** Competition closed September 30, 2026. The best recorded submission scored **0.94646** on the public leaderboard.

## Executive summary

This project developed and submitted a sequence of tabular classification models using LightGBM, CatBoost, XGBoost, and engineered income/commute features. The confirmed public AUC increased from **0.94150** on the baseline to **0.94646** on the final submission, a gain of **0.00496 AUC**. The final recorded submission was #56704749, submitted as `submission_v4.csv`.

The last confirmed ten-fold validation result was **0.946426 AUC**. The submitted blend averaged the log-odds of a four-family XGBoost anchor (50% weight) with two LightGBM families (25% each). Its local validation gain over the anchor was small; the separately measured public leaderboard gain was also small. These are useful comparative results, not evidence of a large or independently established generalization improvement.

## Competition data

The supplied training data contains **668,665 rows**, **13 predictors**, and a binary `Will_Buy_EV` target. The test set contains **286,571 rows**. The training target has 551,886 `No` labels and 116,779 `Yes` labels (17.46% positive). The recorded audit found no missing values, no duplicate training feature rows, and no test-only levels among the audited categorical fields.

The identifier is retained to align output rows but excluded from the principal predictive feature matrices. The dataset has numeric fields such as age, annual income, commute distance, and nearby charging stations, plus categorical fields such as city type, car type, and range-anxiety level. Detailed feature names, input hashes, target counts, and range checks are in `data_audit.json` and `distribution_checks.json`.

Competition CSVs are not included in this repository. Obtain `train.csv`, `test.csv`, and `sample_submission.csv` from Kaggle after accepting the competition rules. A separate public EV-adoption dataset was examined during feature research; it is not part of the selected final model inputs. Its provenance is recorded in `original/provenance.json`. Third-party notebooks informed some experiments; their links and hashes are in `references/provenance.json`, but notebook copies are intentionally not redistributed here.

## Confirmed leaderboard results

| Submission | Submission ID | Public ROC AUC | Snapshot rank | Notes |
|---|---:|---:|---:|---|
| Baseline blend | 56489044 | 0.94150 | Not preserved | Three-fold LightGBM/CatBoost probability blend |
| V2 | 56540777 | 0.94597 | 951 / 2,934 | Additive LightGBM and shallow CatBoost logit blend |
| V3 three-fold | 56619270 | 0.94636 | 689 / 3,215 | Multi-scale and group-feature blend |
| V3 ten-fold | 56655160 | 0.94644 | 500 / 3,352 | Four-family XGBoost blend |
| **V4** | **56704749** | **0.94646** | **511 / 3,496** | **V3 anchor plus standard and linear-leaf LightGBM** |

Ranks are historical snapshots recorded in the submission artifacts. Their team counts differ, so rank changes across rows are not directly comparable. Public AUC is computed on the competition's public test subset; it is not the final private leaderboard score.

## Modeling progression

### Baseline

Three-fold stratified cross-validation (seed 2026) compared LightGBM and CatBoost. The equal probability blend reached **0.941755 OOF AUC**. IDs were excluded as predictors, category levels were learned within each fold for LightGBM, and each training row received an out-of-fold prediction.

### Income-detail model (V2)

The next model retained income at fine resolution through digit-derived income features and constrained interactions. A LightGBM additive model was blended with a shallow, depth-4 GPU CatBoost model in logit space. The three-fold OOF AUC was **0.945698**; the Kaggle public score was **0.94597**.

### Nested target encodings and multi-scale features (V3)

Subsequent experiments added income/commute windows, multi-scale target statistics, and summaries over customers sharing income values or income bins. Inner donor folds were used to calculate target-dependent features without exposing the encoded row's own target. Unlabeled group summaries use predictors only, including test predictors, and do not use test labels.

Features adapted from or motivated by public Kaggle notebooks were implemented and evaluated locally. The project did not use downloaded prediction files. The ten-fold, seed-2026 confirmation fixed iteration counts from earlier screening and blended four XGBoost families, reaching **0.946402 OOF AUC** before submission.

### LightGBM diversity blend (V4)

Standard LightGBM and LightGBM with linear leaves were trained using the rich nested feature view. The final ten-fold blend used:

| Component | Weight |
|---|---:|
| Four-family V3 XGBoost anchor | 50% |
| Standard LightGBM | 25% |
| Linear-leaf LightGBM | 25% |

The two LightGBM families used 900 and 600 fixed trees, respectively. Ten-fold OOF AUC was **0.9464263**. The V3 anchor alone scored **0.9464017** on the same folds, an absolute difference of about 0.000025 AUC. The final CSV scored **0.94646** publicly, compared with **0.94644** for the preceding submission.

Other screened directions included additive logistic models, spline baselines, target encoding at several resolutions, CatBoost variants, asymmetric income bins, and a neural residual model. They were not used in the confirmed final submission when they did not show a sufficiently supported improvement.

## Validation and leakage safeguards

- Three-fold exploratory experiments use stratified folds and record fold-wise as well as pooled AUC.
- Final V3/V4 confirmation uses ten stratified folds with seed 2026 and frozen iteration counts rather than outer-fold early stopping.
- Target encodings and donor-derived margins are built with nested, training-only donors; audit code reconstructs sampled encodings independently.
- Unlabeled train/test group summaries use feature values only, not held-out or test labels.
- Input hashes, fold membership, OOF coverage, model lengths, score records, prediction bounds, submission schema, test-ID ordering, and submitted-file hashes are recorded in metrics, result, and audit artifacts.
- IDs are not used as model features in the principal pipelines; final predictions preserve test ID order.

The same labels were reused for feature, model, and blend selection. Cross-validation scores therefore have selection optimism and overlapping training sets; they are not an untouched holdout. AUC evaluates ranking rather than probability calibration. Small validation gains should be interpreted cautiously, especially after many experiments.

## Reproducibility

Use **Python 3.14** and the pinned dependencies in `requirements_v3.txt`. The repository README has standalone Windows setup instructions, Kaggle data acquisition steps, and the staged commands for rebuilding the experiments. The baseline can be retrained with:

```powershell
.\.venv\Scripts\python.exe train.py
```

This recreates the three-fold baseline and writes generated models and predictions to `outputs/`. The more advanced V2–V4 workflows require the competition inputs and several GB of intermediate predictions, caches, and model files. Run the documented experiment commands sequentially on a machine with adequate memory and disk. CUDA is required for the V2 GPU CatBoost run; the confirmed V4 LightGBM/XGBoost pipeline is CPU-capable.

Large generated models, fold caches, raw data, and intermediate prediction CSVs are excluded from Git to keep the source repository practical. The final V4 submission file, configuration/source code, aggregate metrics, submission-result records, audit scripts, and provenance metadata are retained. Regeneration requires re-running the stages that create upstream checkpoints; it is not a one-command rebuild from a clean clone.

## Selected project files

| File or directory | Purpose |
|---|---|
| `README.md` | Setup, competition overview, full experiment notes, and command reference |
| `REPORT.md` | Final methodology, measured results, validation caveats, and conclusions |
| `train.py`, `improve.py` | Baseline and V2 modeling workflows |
| `boost_v3.py`, `reference_v3.py`, `transductive_v3.py`, `combined_v3.py`, `tenfold_v3.py` | V3 feature construction and model experiments |
| `diversity_v4.py` | V4 LightGBM diversity experiments |
| `audit_v3.py`, `audit_v4.py` | Fold, feature, prediction, and submission checks |
| `select_v2.py`, `select_v3.py`, `select_v4.py` | Candidate selection and submission validation |
| `metrics*.json`, `submission_result*.json` | Frozen scores, parameters, hashes, and leaderboard outcomes |
| `submission_v4.csv` | Best confirmed public submission predictions |
| `data_audit.json`, `distribution_checks.json` | Dataset schema and distribution checks |
| `references/provenance.json` | External notebook attribution and file hashes |

## Conclusion

Careful feature construction and model diversity produced a steady improvement over the initial baseline, with the largest gains arriving from detailed income representations and nested income/commute features. The final submission is a well-documented experimental result, not a claim of first place or an unbiased estimate of future performance. The competition is closed, and the recorded best public score is 0.94646.
