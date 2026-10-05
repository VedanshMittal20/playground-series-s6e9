"""Three-fold probability baselines; run with the workspace virtual environment."""
from pathlib import Path
import hashlib
import json
import time

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 2026


def main():
    started = time.time()
    train = pd.read_csv(ROOT / 'data/train.csv')
    test = pd.read_csv(ROOT / 'data/test.csv')
    sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
    assert set(train.Will_Buy_EV.unique()) == {'Yes', 'No'}
    y = train.Will_Buy_EV.map({'No': 0, 'Yes': 1})
    x = train.drop(columns=['id', 'Will_Buy_EV'])
    xt = test.drop(columns='id')
    assert list(x.columns) == list(xt.columns)
    assert test.id.equals(sample.id)
    assert train.id.is_unique and test.id.is_unique
    assert not train.id.isin(test.id).any()
    cats = x.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cats:
        x[col] = x[col].fillna('__MISSING__').astype(str)
        xt[col] = xt[col].fillna('__MISSING__').astype(str)
    params = {
        'lightgbm': dict(n_estimators=700, learning_rate=0.05, num_leaves=31,
                         min_child_samples=100, colsample_bytree=0.9,
                         reg_lambda=5.0, random_state=SEED, n_jobs=8,
                         verbosity=-1),
        'catboost': dict(iterations=600, learning_rate=0.07, depth=6,
                         loss_function='Logloss', l2_leaf_reg=5,
                         random_seed=SEED, thread_count=8,
                         allow_writing_files=False, verbose=False),
    }
    oof = {k: np.zeros(len(x)) for k in params}
    pred = {k: np.zeros(len(xt)) for k in params}
    folds = np.zeros(len(x), dtype=int)
    scores = {k: [] for k in params}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(cv.split(x, y)):
        folds[va] = fold
        for name in params:
            print(f'Fold {fold+1}/3: {name} training', flush=True)
            a, b, c = x.iloc[tr].copy(), x.iloc[va].copy(), xt.copy()
            if name == 'lightgbm':
                # Learn category vocabularies from this fold only.
                for col in cats:
                    dtype = pd.CategoricalDtype(categories=sorted(a[col].unique()))
                    a[col], b[col], c[col] = (d[col].astype(dtype) for d in (a, b, c))
                model = lgb.LGBMClassifier(**params[name])
                model.fit(a, y.iloc[tr], categorical_feature=cats)
                model.booster_.save_model(str(OUT / f'{name}_fold{fold}.txt'))
            else:
                model = catboost.CatBoostClassifier(**params[name])
                model.fit(a, y.iloc[tr], cat_features=cats)
                model.save_model(str(OUT / f'{name}_fold{fold}.cbm'))
            oof[name][va] = model.predict_proba(b)[:, 1]
            pred[name] += model.predict_proba(c)[:, 1] / 3
            score = roc_auc_score(y.iloc[va], oof[name][va])
            scores[name].append(float(score))
            print(f'Fold {fold+1}/3: {name} AUC={score:.6f}; elapsed={time.time()-started:.0f}s', flush=True)
            pd.DataFrame({'feature': x.columns, 'importance': model.feature_importances_}).to_csv(
                OUT / f'{name}_importance_fold{fold}.csv', index=False)
    oof['blend'] = 0.5 * oof['lightgbm'] + 0.5 * oof['catboost']
    pred['blend'] = 0.5 * pred['lightgbm'] + 0.5 * pred['catboost']
    auc = {name: float(roc_auc_score(y, values)) for name, values in oof.items()}
    selected = max(auc, key=auc.get)
    pd.DataFrame({'id': train.id, 'target': y, 'fold': folds, **oof}).to_csv(OUT / 'oof_predictions.csv', index=False)
    pd.DataFrame({'id': test.id, **pred}).to_csv(OUT / 'test_predictions.csv', index=False)
    for name, values in pred.items():
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        result = sample.copy()
        result['Will_Buy_EV'] = values
        result.to_csv(OUT / f'submission_{name}.csv', index=False)
    chosen = pd.read_csv(OUT / f'submission_{selected}.csv')
    assert list(chosen.columns) == ['id', 'Will_Buy_EV']
    assert chosen.id.equals(test.id) and len(chosen) == len(test)
    chosen.to_csv(ROOT / 'submission.csv', index=False)
    metrics = dict(oof_auc=auc, fold_auc=scores, selected=selected,
                   validation='3 stratified folds, seed 2026; fixed iterations; no early stopping',
                   limitation='Model selection on these folds introduces optimism; no independent final holdout.',
                   prediction_method='Average predictions of the three fold models',
                   parameters=params, categorical_features=cats,
                   train_rows=len(x), test_rows=len(xt), positive_rate=float(y.mean()),
                   seconds=time.time()-started,
                   versions=dict(lightgbm=lgb.__version__, catboost=catboost.__version__,
                                 sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__),
                   input_sha256={n: hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()
                                 for n in ['train.csv', 'test.csv', 'sample_submission.csv']})
    (ROOT / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
