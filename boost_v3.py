"""Cross-fitted multi-scale encodings and GPU XGBoost / CPU LightGBM.

Method inspiration: najiama/pure-lgbm-model-cv-0-94607-lb-0-94638 and
blamerx/s6e9-xgboost-window-encodings-0-946-cv (public Kaggle notebooks).
This implementation learns every label-derived table inside the outer fold.
"""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'experiments_v3'


def features(x):
    d = x.copy()
    for col, factor in [('Annual_Income_USD', 1), ('Daily_Commute_km', 10)]:
        v = np.rint(x[col].to_numpy()*factor).astype(np.int64)
        for k in range(6 if factor == 1 else 3):
            d[f'{col}_digit{k}'] = (v//(10**k))%10
        for width in ([1, 10, 50, 100, 500, 1000, 5000] if factor == 1 else [1, 10, 50]):
            d[f'{col}_key{width}'] = (v//width).astype(str)
    return d


def window_rates(v, y, q, widths, alpha=10):
    axis = np.sort(np.unique(v))
    codes = np.searchsorted(axis, v)
    counts = np.bincount(codes, minlength=len(axis))
    sums = np.bincount(codes, weights=y, minlength=len(axis))
    nc = np.r_[0., np.cumsum(counts)]; ns = np.r_[0., np.cumsum(sums)]
    arrays = []
    for width in widths:
        a = np.searchsorted(axis, q-width, side='left')
        b = np.searchsorted(axis, q+width, side='right')
        n = nc[b]-nc[a]
        arrays.extend([(ns[b]-ns[a]+alpha*np.mean(y))/(n+alpha), np.log1p(n)])
    return np.column_stack(arrays).astype(np.float32)


def encode(a, b, c, y, seed=2026):
    frames = [features(d) for d in [a, b, c]]
    keys = frames[0].astype(str)
    encoder_cols = list(keys)
    extras = [[], [], []]; names = []
    encoders = {}
    for smoothing in [10., 100.]:
        enc = TargetEncoder(smooth=smoothing, cv=5, shuffle=True, random_state=seed, target_type='binary')
        extras[0].append(enc.fit_transform(keys, y).astype(np.float32))
        extras[1].append(enc.transform(frames[1].astype(str)).astype(np.float32))
        extras[2].append(enc.transform(frames[2].astype(str)).astype(np.float32))
        names.extend([f'{name}_te{smoothing:g}' for name in encoder_cols]); encoders[str(smoothing)] = enc
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for col, factor, widths in [('Annual_Income_USD',1,[2,5,10,25,50,200,1000]), ('Daily_Commute_km',10,[1,3,10,30])]:
        values = [np.rint(df[col].to_numpy()*factor).astype(np.int64) for df in [a,b,c]]
        train_features = np.empty((len(a), len(widths)*2), dtype=np.float32)
        for donor, query in splitter.split(a, y):
            train_features[query] = window_rates(values[0][donor], y[donor], values[0][query], widths)
        extras[0].append(train_features)
        for i in [1,2]: extras[i].append(window_rates(values[0],y,values[i],widths))
        names.extend([f'{col}_window{width}_{kind}' for width in widths for kind in ['rate','count']])
    categories = {}
    rawcols = [col for col in frames[0] if '_key' not in col]
    raw = []
    for col in rawcols:
        if pd.api.types.is_string_dtype(frames[0][col]):
            axis = sorted(frames[0][col].unique()); categories[col] = axis
            for df in frames: df[col] = pd.Categorical(df[col], categories=axis).codes.astype(np.float32)
        freq = frames[0][col].value_counts(normalize=True)
        for i,df in enumerate(frames): extras[i].append(df[col].map(freq).fillna(0).to_numpy(np.float32)[:,None])
        names.append(col+'_frequency')
    result = [np.column_stack([df[rawcols].to_numpy(np.float32), *e]) for df,e in zip(frames,extras)]
    return result, rawcols+names, dict(encoders=encoders,categories=categories)


CASES = {
    'xgb_d3': dict(library='xgb', depth=3, rounds=6500, eta=.025, alpha=1., reg_lambda=5., gamma=.05, subsample=.85),
    'xgb_d4': dict(library='xgb', depth=4, rounds=5000, eta=.025, alpha=3., reg_lambda=10., gamma=.1, subsample=.9),
    'xgb_d2': dict(library='xgb', depth=2, rounds=10000, eta=.04, alpha=1., reg_lambda=5., gamma=0., subsample=1.),
    'lgb_d4': dict(library='lgb', depth=4, rounds=7000, eta=.025, alpha=.1, reg_lambda=5., gamma=0., subsample=1.),
}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',default=['xgb_d3']); ap.add_argument('--folds',nargs='+',type=int,default=[0]); args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    train=pd.read_csv(ROOT/'data/train.csv'); test=pd.read_csv(ROOT/'data/test.csv')
    base=pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv'); y=train.Will_Buy_EV.map({'No':0,'Yes':1}).to_numpy()
    assert train.id.equals(base.id) and np.array_equal(y,base.target)
    x=train.drop(columns=['id','Will_Buy_EV']); xt=test.drop(columns='id')
    for fold in args.folds:
        tr=np.flatnonzero(base.fold!=fold); va=np.flatnonzero(base.fold==fold)
        cache=OUT/f'boost_features_fold{fold}.joblib'
        if cache.exists(): matrices,names,state=joblib.load(cache)
        else:
            print(f'Encoding fold {fold}',flush=True)
            matrices,names,state=encode(x.iloc[tr],x.iloc[va],xt,y[tr]); joblib.dump((matrices,names,state),cache,compress=0)
        a,b,c=matrices
        for name in args.models:
            dest=OUT/f'{name}_fold{fold}'
            if dest.with_suffix('.npz').exists(): print(f'Skip {dest.name}',flush=True); continue
            start=time.time(); config=CASES[name]; print(f'Start {dest.name} features={len(names)}',flush=True)
            if config['library']=='xgb':
                params=dict(objective='binary:logistic',eval_metric='auc',device='cuda',tree_method='hist',max_bin=512,
                    max_depth=config['depth'],eta=config['eta'],reg_alpha=config['alpha'],reg_lambda=config['reg_lambda'],
                    gamma=config['gamma'],min_child_weight=5,subsample=config['subsample'],colsample_bytree=.85,nthread=8,seed=2026)
                da=xgb.QuantileDMatrix(a,label=y[tr],max_bin=512,nthread=8,feature_names=names)
                db=xgb.QuantileDMatrix(b,label=y[va],ref=da,max_bin=512,nthread=8,feature_names=names)
                dc=xgb.QuantileDMatrix(c,ref=da,max_bin=512,nthread=8,feature_names=names)
                model=xgb.train(params,da,num_boost_round=config['rounds'],evals=[(db,'validation')],early_stopping_rounds=450,verbose_eval=1000)
                iteration=int(model.best_iteration)+1
                vp=model.predict(db,iteration_range=(0,iteration)); tp=model.predict(dc,iteration_range=(0,iteration))
                model.save_model(str(dest)+'.ubj'); del da,db,dc
            else:
                model=lgb.LGBMClassifier(n_estimators=config['rounds'],learning_rate=config['eta'],max_depth=config['depth'],
                    num_leaves=2**config['depth'],min_child_samples=50,colsample_bytree=.8,reg_alpha=config['alpha'],
                    reg_lambda=config['reg_lambda'],max_bin=512,n_jobs=8,verbosity=-1,random_state=2026)
                model.fit(a,y[tr],eval_set=[(b,y[va])],eval_metric='auc',callbacks=[lgb.early_stopping(450,first_metric_only=True),lgb.log_evaluation(1000)])
                iteration=int(model.best_iteration_); vp=model.predict_proba(b)[:,1]; tp=model.predict_proba(c)[:,1]
                model.booster_.save_model(str(dest)+'.txt')
            assert np.isfinite(vp).all() and np.isfinite(tp).all()
            result=dict(model=name,fold=fold,params=config,iterations=iteration,auc=float(roc_auc_score(y[va],vp)),
                logloss=float(log_loss(y[va],vp)),blend_v2_auc=float(roc_auc_score(y[va],expit((logit(np.clip(vp,1e-7,1-1e-7))+logit(base.equal_logit_blend.iloc[va]))/2))),
                seconds=time.time()-start,features=len(names),validation_note='Outer-fold early stopping introduces selection optimism.')
            np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
            dest.with_suffix('.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
            del model; gc.collect()


if __name__=='__main__': main()
