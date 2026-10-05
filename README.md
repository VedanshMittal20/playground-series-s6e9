# Predicting Electric Vehicle Purchases

Competition: https://www.kaggle.com/competitions/playground-series-s6e9

**Best confirmed result:** ROC AUC **0.94646** (public leaderboard), submission **56704749**. The competition closed on September 30, 2026. See [REPORT.md](REPORT.md) for the complete experiment history, validation approach, results, and caveats.

## Task

- Predict the probability of `Will_Buy_EV` for each test row.
- Evaluation: ROC AUC (higher is better).
- Submission columns: `id,Will_Buy_EV`, preserving test IDs and row order.
- Final deadline in the supplied competition brief: September 30, 2026, 23:59 UTC (October 1, 05:29 IST).

## Data setup

Run commands from this repository's root. Create the pinned Python environment and install dependencies:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_v3.txt
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\kaggle.exe competitions download -c playground-series-s6e9 -p data
Expand-Archive -LiteralPath data/playground-series-s6e9.zip -DestinationPath data -Force
```

First join the competition and accept its rules on Kaggle. Alternatively, place the official `train.csv`, `test.csv`, and `sample_submission.csv` directly in `data/`. Raw data, Kaggle credentials, downloaded reference notebooks, and generated model/cache files are intentionally excluded from Git; provenance and hashes are retained where applicable.

## Initial modeling plan

1. Inspect schema, target balance, missing values, duplicates, and potential grouping or ordering effects before choosing validation splits.
2. Start with stratified cross-validation if the data supports independent row splits. Exclude the ID from baseline features and fit learned preprocessing only on each training fold.
3. Compare CatBoost and LightGBM classifiers using held-out ROC AUC; save out-of-fold probabilities and fold scores.
4. Evaluate a fixed probability blend, then generate test probabilities from the chosen approach.
5. Verify submission columns against the sample, exact test ID alignment, row count, and finite probabilities in [0, 1].

The official competition archive was downloaded and extracted during development on September 23, 2026. Raw CSVs are excluded from Git, so a fresh clone must follow the data setup above. The baseline blend was submitted through the Kaggle CLI on September 23, 2026. Submission **56489044** completed with public ROC AUC **0.94150**; see `submission_result.json`.

The training set has 668,665 rows and 13 predictors; the test set has 286,571 rows. The target uses `Yes` and `No`: encode `Yes` as 1 and predict its probability. Test IDs match the sample submission exactly. See `data_audit.json` for data checks and SHA-256 hashes. Dataset license: CC BY 4.0, as listed on the competition page.

## Baseline results

Three stratified folds (seed 2026), using fixed iteration counts and no early stopping:

| Model | Out-of-fold ROC AUC |
| --- | ---: |
| LightGBM | 0.941586 |
| CatBoost | 0.941554 |
| Equal-weight probability blend | **0.941755** |

`submission.csv` contains the selected blend's 286,571 predictions. Each model's test predictions are averaged across its three fold models, then blended equally. IDs are excluded from features, and LightGBM category vocabularies are learned within each fold. All training rows receive held-out predictions. Selection among these candidates uses the same folds, so this is a local validation estimate with some selection optimism, not an independent holdout or leaderboard score.

Run the baseline from the repository root:

```powershell
.\.venv\Scripts\python.exe train.py
```

`metrics.json` records parameters, versions, fold scores, input hashes, and runtime. `outputs/` contains six saved fold models, feature importances, out-of-fold and test probabilities, and the three candidate submissions. `distribution_checks.json` records category coverage, numeric ranges, and target rates across ID deciles. The data has no missing values or repeated training feature rows; ID-decile target rates range from about 17.14% to 17.57%.

## Improved submission, September 25

`submission_v2.csv` blends an additive LightGBM model with a shallow GPU CatBoost model. The baseline submission remains in `submission.csv`.

Submitted through the Kaggle CLI on September 25, 2026, at 05:16:33 UTC. Submission **56540777** completed with public AUC **0.94597**, up **0.00447** from 0.94150. The 05:17:28 UTC leaderboard snapshot places team **himanshu mittal** at **951 of 2,934**, versus 2,063 immediately before the new submission. Rank is a time-stamped public leaderboard position, not a final private rank. See `submission_result_v2.json` and the downloaded leaderboard snapshot in `leaderboard/`.

| Candidate | Three-fold OOF ROC AUC |
| --- | ---: |
| Additive LightGBM with income digits | 0.945125 |
| CatBoost, depth 4 | 0.945118 |
| Equal probability blend | 0.945646 |
| **Equal logit blend (selected)** | **0.945698** |

The selected blend scores 0.945376, 0.946154, and 0.945563 on the three folds, improving every baseline fold. Its pooled gain over the baseline is 0.003943 AUC. Fold 0 screened model families, feature variants, and iteration counts; the finalists were then trained on folds 1 and 2. These folds also informed final selection, so these figures have selection optimism and are not an independent final holdout.

LightGBM uses 6,000 trees, 16,383 histogram bins, and disjoint interaction groups. Income, its six digits, and its last two/three/four digits share a group; other predictors contribute separately. This preserves fine income information while limiting unrelated interactions. CatBoost uses 3,000 depth-4 trees, exact-value categorical copies of age/income/commute, and three categorical pairs. Full parameters are in `improve.py`. Learned preprocessing is fitted within each training fold; IDs are excluded. GPU training can produce small run-to-run differences.

Each model's test probabilities are averaged over its three fold models. The two averages are combined by averaging their logits and applying the logistic function. Submission checks verify all 286,571 rows, exact ID order, column names, and finite probabilities in [0, 1]. `metrics_v2.json` records the submitted file's SHA-256 and all finalist scores. `experiments_v2/` retains predictions, fold models, and experiment results. The fast additive inference routine checks equivalence against ordinary model inference on dispersed rows.

Reproduce the selected V2 experiment (requires an NVIDIA GPU for CatBoost):

```powershell
.\.venv\Scripts\python.exe improve.py --models additive_digits cat_gpu_shallow --folds 0 1 2
.\.venv\Scripts\python.exe select_v2.py --models additive_digits cat_gpu_shallow
```

The experiment script resumes saved fold predictions. To train a fresh run, use a separate output directory or archive the existing artifacts first. Baseline fold assignments in `outputs/oof_predictions.csv` are required.

Other screened candidates included exact-value LightGBM, cross-fitted target encoding, spline/one-hot logistic regression, additive variants, source-dataset lookup features, and deeper CatBoost. The loose logistic model did not converge and was rejected. The original dataset lookup features did not improve the screening score and are not used in the selected submission. The depth-6 CatBoost/additive blend reached 0.945665, slightly below the selected depth-4 blend.

Income features and high-resolution/additive modeling were informed by the public [feature ablation discussion](https://www.kaggle.com/competitions/playground-series-s6e9/discussion/741755). All selected models were trained locally from the competition training labels; no public prediction files were used. The separately tested [original EV dataset](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety) has its provenance recorded in `original/provenance.json`.

## Third modeling round, September 28–29

Submission **56619270** (`submission_v3_threefold.csv`) scored **0.94636 public AUC**, improving the prior 0.94597. The leaderboard snapshot at 2026-09-27 20:59 UTC (September 28 in India) placed the team **689 / 3,215**. Its three-fold OOF AUC is **0.9461972355**. Exact submitted-file hashes and results are stored in `metrics_v3_threefold.json` and `submission_result_v3_threefold.json`.

The new models combine income/commute target statistics at several resolutions, local window rates, and predictor-only summaries of customers sharing an income value or income bin. Target encodings use inner donor folds within each outer training fold. The unlabeled summaries include train and test predictors, but never their labels. IDs are excluded. A second feature view adds regularized additive coordinates and composition adjustments from [Paul Bryan Elefante's public XGBoost notebook](https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample), reproduced locally on our folds and extended with our feature views. No downloaded prediction files are used.

| Model / blend | Three-fold OOF AUC |
| --- | ---: |
| Multi-scale XGBoost | 0.945856 |
| Multi-scale LightGBM | 0.945835 |
| Reproduced composition model | 0.945975 |
| XGBoost plus unlabeled group summaries | 0.946034 |
| Combined feature model | 0.946124 |
| Equal logit blend of the last two (submitted) | **0.946197** |

Exact-value penalized logistic models, neighbor smoothness, spline baselines, and a low-cardinality interaction were also tested. They did not beat the selected models. More detailed unlabeled summaries produced a small three-fold gain and are treated as an exploratory variant, not evidence of a large improvement.

The final confirmation uses ten stratified folds (seed 2026), allowing each model to train on 90% of the competition rows. Training lengths are frozen from the earlier experiments: 900 trees for the reference model, 1,500 for the unlabeled-group model, and 700 for the combined feature model. The three-fold screening used outer-fold early stopping; the ten-fold runs do not. Method and blend choices still reuse training labels, so neither estimate is an independent final holdout. Any reported paired AUC interval treats fitted predictions as fixed and omits selection effects and dependence between CV training sets.

Reproduce the three-fold screening and ten-fold confirmation from the repository root:

```powershell
.\.venv\Scripts\python.exe boost_v3.py --models xgb_d4 lgb_d4 --folds 0 1 2
.\.venv\Scripts\python.exe reference_v3.py --folds 0 1 2
.\.venv\Scripts\python.exe transductive_v3.py --folds 0 1 2
.\.venv\Scripts\python.exe combined_v3.py --folds 0 1 2
.\.venv\Scripts\python.exe reference_v3.py --scheme ten --iterations 900 --folds 0 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe tenfold_v3.py --models xgb_unlabeled xgb_combined --folds 0 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe select_v3.py --scheme ten --stem new_run
```

The scripts resume completed folds. `select_v3.py` verifies complete OOF coverage, exact fold indices, finite predictions, submission columns, and test ID order. It refuses to overwrite an artifact with a recorded submission result. Feature caches and saved models require several GB of disk space; do not remove them while training is running. `requirements_v3.txt` pins the environment. `references/provenance.json` records downloaded notebook hashes and attribution; `GOAL_PROGRESS.md` records the working state.

Multi-scale/window encodings were informed by [Naji's LightGBM notebook](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94607-lb-0-94638) and [BlamerX's window-encoding notebook](https://www.kaggle.com/code/blamerx/s6e9-xgboost-window-encodings-0-946-cv). `heuljax_features.py` is an attributed adaptation of Elefante's feature code. The remaining experiment and blending scripts were implemented for this project.

### Completed ten-fold submission, September 29

Submission **56655160**, `submission_v3.csv`, scored **0.94644 public AUC**. The leaderboard at 2026-09-28 23:59:45 UTC (September 29 in India) placed the team **500 / 3,352**, up from **733 / 3,352** immediately before submission. This is a current public rank, not the final competition result. `submission_result_v3.json` preserves the score, timestamp, rank, and exact file hash.

| Ten-fold model / blend | OOF AUC |
| --- | ---: |
| Reproduced composition model | 0.946256 |
| Unlabeled group model | 0.946220 |
| Combined model | 0.946346 |
| Combined model with richer group summaries | 0.946354 |
| **Equal logit blend of all four (submitted)** | **0.946402** |

Each family averages its ten test-prediction vectors; the four family averages receive equal logit weights. The richer combined variant also uses 700 fixed iterations. `metrics_v3.json` records the complete small blend menu, full-precision scores, all model parameters, and validation caveats. Gains between the best blends are very small and should not be interpreted as independent statistical evidence.

`audit_v3.py --require-complete` independently reconstructs sampled income target encodings from permitted donor rows, verifies all ten folds, checks labels and IDs against the unchanged input hashes, and checks prediction indices and bounds. The saved CSV also passed a read-back check for exact test IDs, schema, row count, finite probabilities, and SHA-256. Audit results are in `experiments_v3/tenfold/audit.json`.

To reproduce the final richer variant and selection after the earlier training commands:

```powershell
.\.venv\Scripts\python.exe tenfold_v3.py --models xgb_combined_rich --folds 0 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe audit_v3.py --require-complete
.\.venv\Scripts\python.exe select_v3.py --scheme ten --models heuljax_xgb xgb_unlabeled xgb_combined xgb_combined_rich --stem reproduced_v3
```

## LightGBM diversity round, September 30

Submission **56704749**, `submission_v4.csv`, scored **0.94646 public AUC**. The 2026-09-30 10:28:00 UTC leaderboard placed the team **511 / 3,496**, versus **571 / 3,496** before submission. It improves the preceding 0.94644. The exact submitted-file hash, score and time-stamped rank are in `submission_result_v4.json`.

| Ten-fold candidate | OOF AUC |
| --- | ---: |
| Submitted V3 XGBoost anchor | 0.946401727 |
| Standard LightGBM | 0.946402646 |
| LightGBM with linear leaves | 0.946390437 |
| 50% anchor + 50% standard LightGBM | 0.946423198 |
| **50% anchor + 25% each LightGBM (submitted)** | **0.946426264** |

The LightGBM families use the same rich nested feature view and donor-only initial margins, with 900 and 600 fixed trees. Each family averages ten test probabilities before the weighted logit blend. The V3 anchor contains four XGBoost families, so the final file combines 60 fold models. `metrics_v4_ten.json` records the small fixed blend menu, full parameters, the exact preserved anchor, input hashes and validation limitations. These small gains reuse labels for method selection; the approximate paired interval does not include selection or overlapping CV training.

`audit_v4.py` verifies all twenty LightGBM model lengths, complete held-out row coverage, fold indices, probabilities, recorded AUCs, unchanged inputs and the final CSV. It also reloads both fold-0 models, reapplies their saved preprocessing and initial margin, and reproduces predictions on 120 dispersed test rows. Results are in `experiments_v4/ten/audit.json`.

The [Single Model - Zoom Zoom notebook](https://www.kaggle.com/code/jazivxt/single-model-zoom-zoom) suggested trying linear leaves. Our implementation retains the project's nested feature construction and validation. No external prediction files were downloaded or used. CatBoost's blended three-fold gain was negligible and it was dropped. The asymmetric income-bin variant also failed to add a useful fold-0 gain and was dropped. Neural residual modeling is being screened separately in `experiments_v6/`.

Reproduce the final diversity run after generating the V3 feature caches:

```powershell
.\.venv\Scripts\python.exe diversity_v4.py --scheme ten --models lgb_combined --iterations 900 --folds 0 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe diversity_v4.py --scheme ten --models lgb_linear --iterations 600 --folds 0 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe select_v4.py --scheme ten --models lgb_combined lgb_linear --stem reproduced_v4 --write-submission
.\.venv\Scripts\python.exe audit_v4.py
```

Run the two training commands sequentially on this 16 GB machine. Read-only memory-mapped caches and blockwise scaling limit peak memory. The scripts resume completed folds. `final_selection_check.json` records that Kaggle's automatic final selection was enabled; the site selects up to two best-scoring submissions when fewer than two are manually selected. The verified deadline is September 30, 2026 at 23:59 UTC (October 1 at 05:29 IST).
