# Goal: strongest validated EV prediction solution, aiming for first place

> **Archived:** The competition closed on September 30, 2026. First place was not achieved. This file preserves the experiment log as it stood during the competition; the final confirmed result is submission #56704749 with public AUC 0.94646. See `REPORT.md` for the final summary.

Started September 28, 2026. A first-place finish is an aspiration, not a promised outcome. The user requested continued work on September 29. The latest goal-tracker check reports `active`.

## Confirmed submission before this round

- Submission 56540777: public ROC AUC 0.94597; three-fold OOF 0.9456975802.
- Latest downloaded board at 2026-09-27 20:34 UTC: rank 1,078 / 3,215.
- Leader 0.94945, second 0.94689. The large first-place gap is unexplained publicly; comments are not proof of a leak or misconduct.

## Current experiment round

All new artifacts are isolated in `experiments_v3/`. Baseline folds and submissions are preserved.

1. Exact-value penalized additive logistic models (`gam_v3.py`): tested ridge and neighborhood smoothness, spline baselines, and a low-cardinality interaction. All converged; none improved the previous blend enough to retain.
2. Nested multi-scale target encodings and income/commute window rates (`boost_v3.py`): XGBoost and LightGBM confirmed improvements on all three folds.
3. Reproduced Paul Bryan Elefante's public feature method (`heuljax_features.py`, `reference_v3.py`) on our saved folds. Source attribution and downloaded notebook hashes are in `references/provenance.json`. No downloaded predictions were used. OOF AUC: 0.9459747616.
4. Predictor-only train/test income-group means (`transductive_v3.py`) improved the new XGBoost across all three folds. These aggregates do not access validation or test labels.
5. Combined feature views (`combined_v3.py`) completed all three folds: OOF 0.9461237008. Equal logit blending with the unlabeled-group model improved OOF to 0.9461972355.
6. The three-fold blend was submitted as 56619270: public AUC **0.94636**, rank **689 / 3,215** at 2026-09-27 20:59 UTC. The file and result are preserved under `submission_v3_threefold.csv` and `submission_result_v3_threefold.json`.
7. The reference model completed all ten folds with 900 fixed iterations: OOF **0.9462556902**. `metrics_v3_reference.json` records it; this standalone candidate has not been submitted.
8. The other ten-fold models stopped after folds 0 and 1 in the prior run. On September 29, `tenfold_v3.py --models xgb_unlabeled xgb_combined --folds 2 3 4 5 6 7 8 9` resumed the missing folds. Training lengths are fixed at 1,500 and 700 iterations respectively.
9. Richer predictor-only group summaries completed three-fold testing: the combined model reached 0.9461435848 and its equal logit blend with the unlabeled model reached 0.9462097554. The gain was small, so the variant was also checked across all ten final folds.
10. All 40 ten-fold models completed. OOF AUC: reference 0.9462556902, unlabeled 0.9462200551, combined 0.9463462045, rich combined 0.9463540175. A small fixed blend menu selected the equal logit blend of all four, OOF **0.9464017269**. `audit_v3.py --require-complete` passed for all ten folds, including independent reconstruction of sampled inner and outer target encodings.
11. Submission **56655160**, `submission_v3.csv`, scored **0.94644**. The 2026-09-28 23:59:45 UTC snapshot places the team **500 / 3,352**, versus **733 / 3,352** immediately before this submission. Submitted file, hash, metrics, source hashes, and scored result are preserved.
12. `diversity_v4.py` completed three-fold screens using the rich nested features and donor-only initial margins. CatBoost added only 0.000001466 blended AUC and was rejected. Standard and linear-leaf LightGBM each improved all three folds when blended. The fixed small menu selected 50% of the previous rich XGB blend plus 25% of each LightGBM model, OOF **0.9462468593** versus anchor **0.9462097554**. Full metrics are in `metrics_v4_screen_three.json`.
13. Ten-fold confirmation freezes 900 standard-LightGBM trees and 600 linear-leaf trees. On September 30 both old process handles were missing, no Python process was live, and only standard fold 0 had completed. The missing folds were resumed sequentially, using memory-mapped caches and smaller scaling temporaries. Model parameters and nested feature definitions remain unchanged. `experiments_v4/plan.json` records the screen, fixed lengths, and authoritative resume checks.
14. At 2026-09-30 05:23:53 UTC, submission 56655160 remained the latest, with public AUC **0.94644**, rank **555 / 3,470**. The first-place score remained 0.94945; second was 0.94697. The competition deadline is September 30, 23:59 UTC.
15. Both LightGBM families completed all ten folds. Standard LightGBM OOF is 0.9464026457, linear-leaf LightGBM 0.9463904370. The selected blend uses 50% of the V3 anchor and 25% of each LightGBM family, OOF **0.9464262638**. `audit_v4.py` checked all twenty model lengths, prediction arrays and AUCs, unchanged input hashes, the submission, and reloaded model inference on 120 dispersed test rows for both families. All passed.
16. Submission **56704749**, `submission_v4.csv`, scored **0.94646**. At 2026-09-30 10:28:00 UTC the rank is **511 / 3,496**, versus **571 / 3,496** immediately before submission. The submitted file and all earlier submissions are preserved. `submission_result_v4.json` records the exact hash and scored result. The final-selection page was verified in the browser: zero manual selections, automatic best-scoring selection of up to two submissions.
17. Asymmetric nearby-income bin features passed direct own-label exclusion and held-out formula checks. The XGBoost screen reached 0.945803 on fold 0; quarter blending added only 0.000007045, so it was rejected for full training. The code and screening result remain as reproducible evidence under `asymmetric_features_v5.py` and `experiments_v4/three/xgb_asymmetric_fold0.json`.
18. `residual_mlp_v6.py` screens a small neural residual model on nested donor-only features and initial margins. All parameter groups passed numerical gradient checks. Training-only scaling and category vocabularies are used; a fixed cosine schedule is retained when a final epoch count is frozen. Fold 0 is running under `experiments_v6/three/`.

## Validation interpretation

Every OOF row is predicted by a model that excludes that row's label. Inner donor folds protect target encodings and initial margins. Three-fold experiments use outer-fold early stopping; final ten-fold fits use frozen training lengths. These are reused labels for method selection, so scores still have selection optimism and are not an independent final holdout.

The first four-model three-fold blend improved OOF AUC to 0.9460865425. Its paired AUC gain over v2 is 0.0003889623; the approximate fixed-prediction 95% interval is [0.0003429403, 0.0004349842], which does not account for model selection or overlapping CV training. Newer feature views may replace this candidate before submission.

## Next actions

Finish the neural screen, confirm any useful gain on the other folds, and continue searching for methods that improve validation enough to justify submission before the deadline. Preserve reproducible parameters, dependencies, source attribution, and the experiment record. First place has not been achieved; the goal remains active.
