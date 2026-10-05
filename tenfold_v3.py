"""Fixed-budget ten-fold confirmation of the two new feature views."""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score,log_loss
from boost_v3 import encode
from transductive_v3 import unlabeled_features
import heuljax_features as hf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v3'/'tenfold'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',nargs='+',type=int,default=list(range(10)))
    ap.add_argument('--models',nargs='+',choices=['xgb_unlabeled','xgb_combined','xgb_combined_rich'],default=['xgb_unlabeled'])
    ap.add_argument('--wait-for-cache',action='store_true')
    args=ap.parse_args()
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    base=pd.read_csv(OUT/'folds.csv');y=hf.y
    assert train.id.equals(base.id) and np.array_equal(base.id,hf.TRAIN_IDS) and np.array_equal(base.target,y)
    x=train.drop(columns=['id','Will_Buy_EV']);xt=test.drop(columns='id')
    rich='xgb_combined_rich' in args.models
    if rich and len(args.models)!=1:raise ValueError('Run the rich feature variant separately')
    ux,ut,un=unlabeled_features(x,xt,rich=rich)
    for fold in args.folds:
        if all((OUT/f'{name}_fold{fold}.npz').exists() for name in args.models):continue
        tr=np.flatnonzero(base.fold!=fold);va=np.flatnonzero(base.fold==fold)
        cache=OUT/f'boost_features_fold{fold}.joblib'
        if rich and args.wait_for_cache:
            ready=OUT/f'xgb_unlabeled_fold{fold}.json'
            deadline=time.monotonic()+600
            if not ready.exists():print(f'Waiting for completed feature cache {fold}',flush=True)
            while not ready.exists() and time.monotonic()<deadline:time.sleep(5)
            if not ready.exists():raise TimeoutError(f'Standard feature generation did not complete for fold {fold}')
        if rich and not cache.exists():raise FileNotFoundError('Complete the standard feature cache first')
        if cache.exists():matrices,names,state=joblib.load(cache)
        else:
            print(f'Encode ten-fold {fold}',flush=True);matrices,names,state=encode(x.iloc[tr],x.iloc[va],xt,y[tr]);joblib.dump((matrices,names,state),cache,compress=0)
        for name in args.models:
            dest=OUT/f'{name}_fold{fold}'
            if dest.with_suffix('.npz').exists():continue
            start=time.time();print(f'Start ten-fold {dest.name}',flush=True)
            params=dict(objective='binary:logistic',eval_metric='auc',device='cuda',tree_method='hist',max_bin=512,max_depth=4,eta=.025,reg_alpha=3.,reg_lambda=10.,gamma=.1,min_child_weight=5,subsample=.9,colsample_bytree=.85,nthread=8,seed=2026+fold)
            data=[np.column_stack([d,u]) for d,u in zip(matrices,[ux[tr],ux[va],ut])]
            cols=names+un;types=['q']*len(cols);margins=[None,None,None];iterations=1500
            if name in ['xgb_combined','xgb_combined_rich']:
                source_cache=OUT/f'heuljax_features_fold{fold}.joblib'
                if not source_cache.exists():raise FileNotFoundError(f'Complete reference feature fold {fold} first')
                ra,rb,rc,*margins=joblib.load(source_cache)
                data=[np.column_stack([d,e]) for d,e in zip([ra,rb,rc],data)]
                cols=hf.FEATURE_COLS+['additional_'+n for n in names]+un
                types=hf.FEATURE_TYPES+['q']*(len(names)+len(un))
                params=dict(hf.XGB_PARAMS,max_depth=4,learning_rate=.025,reg_alpha=5.,reg_lambda=15.,colsample_bynode=.8,seed=2026+fold)
                params['monotone_constraints']=params['monotone_constraints'][:-1]+',0'*(len(names)+len(un))+')'
                iterations=700
                del ra,rb,rc
            dm=[]
            for i,(d,m) in enumerate(zip(data,margins)):
                dm.append(xgb.QuantileDMatrix(d,label=y[tr] if i==0 else (y[va] if i==1 else None),base_margin=m,
                    max_bin=params['max_bin'],nthread=8,feature_names=cols,feature_types=types,enable_categorical=True,ref=None if i==0 else dm[0]))
            model=xgb.train(params,dm[0],num_boost_round=iterations,evals=[(dm[1],'validation')],verbose_eval=500)
            vp=model.predict(dm[1]);tp=model.predict(dm[2]);assert np.isfinite(vp).all() and np.isfinite(tp).all()
            model.save_model(str(dest)+'.ubj');np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
            result=dict(model=name,scheme='ten',fold=fold,params=params,iterations=iterations,auc=float(roc_auc_score(y[va],vp)),logloss=float(log_loss(y[va],vp)),seconds=time.time()-start,
                validation_note='Fixed training length; shared training data with earlier selection so not a fresh holdout. Nested target encodings; unlabeled train/test predictor aggregates.')
            dest.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='params'}),flush=True)
            del data,dm,model,margins;gc.collect()
        del matrices,names,state;gc.collect()


if __name__=='__main__':main()
