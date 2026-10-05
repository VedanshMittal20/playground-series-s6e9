"""Combine donor-only feature views with unlabeled income-group summaries."""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from transductive_v3 import unlabeled_features
import heuljax_features as hf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v3'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',nargs='+',type=int,default=[0]);ap.add_argument('--rich',action='store_true');args=ap.parse_args()
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    base=pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv');y=hf.y
    assert train.id.equals(base.id) and np.array_equal(train.id,hf.TRAIN_IDS) and np.array_equal(y,base.target)
    ux,ut,un=unlabeled_features(train.drop(columns=['id','Will_Buy_EV']),test.drop(columns='id'),rich=args.rich)
    name='xgb_combined_rich' if args.rich else 'xgb_combined'
    for fold in args.folds:
        dest=OUT/f'{name}_fold{fold}'
        if dest.with_suffix('.npz').exists():continue
        start=time.time();tr=np.flatnonzero(base.fold!=fold);va=np.flatnonzero(base.fold==fold)
        a,b,c,am,bm,cm=joblib.load(OUT/f'heuljax_features_fold{fold}.joblib')
        matrices,extra_names,_=joblib.load(OUT/f'boost_features_fold{fold}.joblib')
        names=hf.FEATURE_COLS+['additional_'+n for n in extra_names]+un
        types=hf.FEATURE_TYPES+['q']*(len(extra_names)+len(un))
        a,b,c=[np.column_stack([d,e,u]) for d,e,u in zip([a,b,c],matrices,[ux[tr],ux[va],ut])]
        params=dict(hf.XGB_PARAMS,max_depth=4,learning_rate=.025,reg_alpha=5.,reg_lambda=15.,colsample_bynode=.8,seed=2026)
        params['monotone_constraints']=params['monotone_constraints'][:-1]+',0'*(len(extra_names)+len(un))+')'
        da=xgb.QuantileDMatrix(a,label=y[tr],base_margin=am,max_bin=256,nthread=8,feature_names=names,feature_types=types,enable_categorical=True)
        db=xgb.QuantileDMatrix(b,label=y[va],base_margin=bm,max_bin=256,nthread=8,feature_names=names,feature_types=types,enable_categorical=True,ref=da)
        dc=xgb.QuantileDMatrix(c,base_margin=cm,max_bin=256,nthread=8,feature_names=names,feature_types=types,enable_categorical=True,ref=da)
        model=xgb.train(params,da,num_boost_round=4000,evals=[(db,'validation')],early_stopping_rounds=450,verbose_eval=500)
        trees=model.best_iteration+1;vp=model.predict(db,iteration_range=(0,trees));tp=model.predict(dc,iteration_range=(0,trees))
        model.save_model(str(dest)+'.ubj');np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
        result=dict(model=name,fold=fold,params=params,iterations=trees,auc=float(roc_auc_score(y[va],vp)),blend_v2_auc=float(roc_auc_score(y[va],expit((logit(vp)+logit(base.equal_logit_blend.iloc[va]))/2))),seconds=time.time()-start,features=len(names),validation_note='Donor-only target encodings, transductive predictor-only group moments, outer-fold early stopping.')
        dest.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        del model,a,b,c,da,db,dc,matrices;gc.collect()


if __name__=='__main__':main()
