"""Evaluate the public heuljax feature method on this project's saved folds."""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import heuljax_features as hf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v3'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--folds',nargs='+',type=int,default=[0])
    ap.add_argument('--scheme',choices=['three','ten'],default='three')
    ap.add_argument('--iterations',type=int,default=0)
    args=ap.parse_args()
    base=pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv')
    assert np.array_equal(base.id,hf.TRAIN_IDS) and np.array_equal(base.target,hf.y)
    out=OUT if args.scheme=='three' else OUT/'tenfold'
    out.mkdir(exist_ok=True)
    if args.scheme=='ten':
        assignments=np.zeros(len(base),dtype=int)
        for f,(_,idx) in enumerate(StratifiedKFold(10,shuffle=True,random_state=2026).split(base.id,hf.y)): assignments[idx]=f
        base['fold']=assignments
        base[['id','target','fold']].to_csv(out/'folds.csv',index=False)
    for fold in args.folds:
        dest=out/f'heuljax_xgb_fold{fold}'
        if dest.with_suffix('.npz').exists(): print(f'Skip {dest.name}',flush=True); continue
        start=time.time(); tr=np.flatnonzero(base.fold!=fold); va=np.flatnonzero(base.fold==fold)
        cache=out/f'heuljax_features_fold{fold}.joblib'
        with threadpool_limits(limits=8):
            if cache.exists(): bundle=joblib.load(cache)
            else:
                print(f'Building donor-only features fold {fold}',flush=True)
                bundle=hf.build_fold_bundle(tr,va,fold); joblib.dump(bundle,cache,compress=0)
            a,b,c,am,bm,cm=bundle
            params=dict(hf.XGB_PARAMS,seed=2026+fold if args.scheme=='ten' else 2026)
            da=xgb.QuantileDMatrix(a,label=hf.y[tr],base_margin=am,feature_names=hf.FEATURE_COLS,feature_types=hf.FEATURE_TYPES,enable_categorical=True,max_bin=256,nthread=8)
            db=xgb.QuantileDMatrix(b,label=hf.y[va],base_margin=bm,feature_names=hf.FEATURE_COLS,feature_types=hf.FEATURE_TYPES,enable_categorical=True,max_bin=256,nthread=8,ref=da)
            dc=xgb.QuantileDMatrix(c,base_margin=cm,feature_names=hf.FEATURE_COLS,feature_types=hf.FEATURE_TYPES,enable_categorical=True,max_bin=256,nthread=8,ref=da)
            print(f'Training {dest.name}',flush=True)
            model=xgb.train(params,da,num_boost_round=args.iterations or 6000,evals=[(db,'validation')],early_stopping_rounds=None if args.iterations else 750,verbose_eval=500)
            trees=args.iterations or model.best_iteration+1
            vp=model.predict(db,iteration_range=(0,trees)); tp=model.predict(dc,iteration_range=(0,trees))
            assert np.isfinite(vp).all() and np.isfinite(tp).all()
            model.save_model(str(dest)+'.ubj')
            np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
            result=dict(model='heuljax_xgb',scheme=args.scheme,fold=fold,params=params,iterations=trees,auc=float(roc_auc_score(hf.y[va],vp)),
                logloss=float(log_loss(hf.y[va],vp)),blend_v2_auc=float(roc_auc_score(hf.y[va],expit((logit(np.clip(vp,1e-7,1-1e-7))+logit(base.equal_logit_blend.iloc[va]))/2))),seconds=time.time()-start,
                source='https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample',validation_note=('Fixed iterations selected from prior three-fold experiments; reused training labels mean this is not an independent holdout.' if args.iterations else 'Outer-fold early stopping introduces selection optimism.')+' Original source data is external training data; nested donor folds exclude validation labels.')
            dest.with_suffix('.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
            del bundle,a,b,c,am,bm,cm,da,db,dc,model; gc.collect()


if __name__=='__main__': main()
