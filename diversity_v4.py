"""Screen different learners on existing nested features and donor-only margins."""
import argparse
import gc
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from transductive_v3 import unlabeled_features
import heuljax_features as hf

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--models',nargs='+',choices=['lgb_combined','lgb_linear','cat_combined','xgb_asymmetric'],required=True)
    ap.add_argument('--folds',nargs='+',type=int,default=[0])
    ap.add_argument('--scheme',choices=['three','ten'],default='three')
    ap.add_argument('--iterations',type=int,default=0)
    args=ap.parse_args()
    if 'xgb_asymmetric' in args.models and len(args.models)!=1:
        raise ValueError('Run asymmetric features separately to keep feature matrices isolated')
    cache=ROOT/'experiments_v3'
    cache=cache/'tenfold' if args.scheme=='ten' else cache
    out=ROOT/'experiments_v4'/args.scheme;out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(cache/'folds.csv' if args.scheme=='ten' else ROOT/'experiments_v2/oof_predictions.csv')
    assert np.array_equal(base.id,hf.TRAIN_IDS) and np.array_equal(base.target,hf.y)
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    ux,ut,un=unlabeled_features(train.drop(columns=['id','Will_Buy_EV']),test.drop(columns='id'),rich=True)
    for f in args.folds:
        if all((out/f'{n}_fold{f}.npz').exists() and (out/f'{n}_fold{f}.json').exists() for n in args.models):
            print(f'Skip completed {args.scheme} fold {f}',flush=True)
            continue
        tr=np.flatnonzero(base.fold!=f);va=np.flatnonzero(base.fold==f)
        ra,rb,rc,am,bm,cm=joblib.load(cache/f'heuljax_features_fold{f}.joblib',mmap_mode='r')
        matrices,extra_names,_=joblib.load(cache/f'boost_features_fold{f}.joblib',mmap_mode='r')
        names=hf.FEATURE_COLS+['additional_'+n for n in extra_names]+un
        data=[np.column_stack([r,e,u]) for r,e,u in zip([ra,rb,rc],matrices,[ux[tr],ux[va],ut])]
        cats=[i for i,t in enumerate(hf.FEATURE_TYPES) if t=='c']
        del ra,rb,rc,matrices;gc.collect()
        previous=np.load(cache/f'xgb_combined_rich_fold{f}.npz')
        unlabeled=np.load(cache/f'xgb_unlabeled_fold{f}.npz')
        np.testing.assert_array_equal(previous['indices'],va)
        anchor=expit((logit(np.clip(previous['validation'],1e-7,1-1e-7))+logit(np.clip(unlabeled['validation'],1e-7,1-1e-7)))/2)
        for name in args.models:
            dest=out/f'{name}_fold{f}'
            if dest.with_suffix('.npz').exists() and dest.with_suffix('.json').exists():continue
            start=time.time();print(f'Training {name} {args.scheme} fold {f}',flush=True)
            if name=='xgb_asymmetric':
                import xgboost as xgb
                from asymmetric_features_v5 import encode_asymmetric
                more,new_names=encode_asymmetric(train.Annual_Income_USD.iloc[tr],
                    train.Annual_Income_USD.iloc[va],test.Annual_Income_USD,hf.y[tr])
                for i in range(3):data[i]=np.column_stack([data[i],more[i]])
                del more
                names=names+new_names
                types=hf.FEATURE_TYPES+['q']*(len(names)-len(hf.FEATURE_TYPES))
                params=dict(hf.XGB_PARAMS,max_depth=4,learning_rate=.025,reg_alpha=5.,reg_lambda=15.,seed=2026+f)
                params['monotone_constraints']=params['monotone_constraints'][:-1]+',0'*(len(names)-len(hf.FEATURE_TYPES))+')'
                dm=[]
                for i,(d,m) in enumerate(zip(data,[am,bm,cm])):
                    dm.append(xgb.QuantileDMatrix(d,label=hf.y[tr] if i==0 else (hf.y[va] if i==1 else None),
                        base_margin=m,max_bin=256,nthread=8,feature_names=names,feature_types=types,
                        enable_categorical=True,ref=None if i==0 else dm[0]))
                model=xgb.train(params,dm[0],num_boost_round=args.iterations or 2500,
                    evals=[(dm[1],'validation')],early_stopping_rounds=None if args.iterations else 400,verbose_eval=500)
                iterations=args.iterations or model.best_iteration+1
                vp=model.predict(dm[1],iteration_range=(0,iterations))
                tp=model.predict(dm[2],iteration_range=(0,iterations))
                expected=None if args.iterations else float(model.best_score)
                model.save_model(str(dest)+'.ubj')
                del dm
            elif name in ['lgb_combined','lgb_linear']:
                import lightgbm as lgb
                params=dict(objective='binary',metric='auc',learning_rate=.025,max_depth=4,num_leaves=15,
                    min_data_in_leaf=200,lambda_l1=5.,lambda_l2=15.,feature_fraction=.85,
                    bagging_fraction=.9,bagging_freq=1,max_bin=255,num_threads=8,verbosity=-1,
                    seed=2026+f,boost_from_average=False)
                current=data
                if name=='lgb_linear':
                    # Linear leaves suggested by jazivxt's public Single Model -
                    # Zoom Zoom notebook; fit scaling only on this outer donor set.
                    # https://www.kaggle.com/code/jazivxt/single-model-zoom-zoom
                    params.update(linear_tree=True,linear_lambda=10.,feature_fraction=.3,
                                  min_data_in_leaf=100,learning_rate=.03)
                    numerical=np.array([i for i in range(len(names)) if i not in cats])
                    # Small column blocks avoid multi-GB temporary arrays.
                    mean=np.empty(len(numerical),dtype=data[0].dtype)
                    scale=np.empty_like(mean)
                    for start_col in range(0,len(numerical),32):
                        stop_col=min(start_col+32,len(numerical))
                        block=data[0][:,numerical[start_col:stop_col]]
                        mean[start_col:stop_col]=np.nanmean(block,axis=0)
                        scale[start_col:stop_col]=np.nanstd(block,axis=0)
                    del block
                    scale[scale<1e-6]=1
                    current=data if len(args.models)==1 else [d.copy() for d in data]
                    center=np.zeros(len(names),dtype=mean.dtype);spread=np.ones_like(center)
                    center[numerical]=mean;spread[numerical]=scale
                    for d in current:
                        d-=center
                        d/=spread
                    np.savez(dest.with_name(dest.name+'_scaler').with_suffix('.npz'),columns=numerical,mean=mean,scale=scale)
                da=lgb.Dataset(current[0],label=hf.y[tr],init_score=am,feature_name=names,categorical_feature=cats)
                db=lgb.Dataset(current[1],label=hf.y[va],init_score=bm,reference=da,feature_name=names,categorical_feature=cats)
                callbacks=[lgb.log_evaluation(500)]
                if not args.iterations:callbacks.append(lgb.early_stopping(400))
                model=lgb.train(params,da,num_boost_round=args.iterations or (2000 if name=='lgb_linear' else 4000),valid_sets=[db],callbacks=callbacks)
                iterations=args.iterations or model.best_iteration
                vp=expit(model.predict(current[1],raw_score=True,num_iteration=iterations)+bm)
                tp=expit(model.predict(current[2],raw_score=True,num_iteration=iterations)+cm)
                model.save_model(str(dest)+'.txt',num_iteration=iterations)
                expected=model.best_score['valid_0']['auc']
                del da,db,current
            else:
                from catboost import CatBoostClassifier,Pool
                frames=[pd.DataFrame(d,columns=names) for d in data]
                for frame in frames:
                    for col in cats:frame[names[col]]=frame[names[col]].fillna(-1).astype(np.int32).astype(str)
                da=Pool(frames[0],label=hf.y[tr],baseline=am,cat_features=cats)
                db=Pool(frames[1],label=hf.y[va],baseline=bm,cat_features=cats)
                params=dict(iterations=args.iterations or 4000,depth=4,learning_rate=.025,l2_leaf_reg=15,
                    loss_function='Logloss',eval_metric='AUC',task_type='GPU',devices='0',thread_count=8,
                    random_seed=2026+f,one_hot_max_size=20,verbose=500,allow_writing_files=False)
                model=CatBoostClassifier(**params)
                model.fit(da,eval_set=db,early_stopping_rounds=None if args.iterations else 400,use_best_model=not bool(args.iterations))
                iterations=model.tree_count_
                # Predict without a Pool baseline and add the donor-only margin once.
                vp=expit(model.predict(frames[1],prediction_type='RawFormulaVal')+bm)
                tp=expit(model.predict(frames[2],prediction_type='RawFormulaVal')+cm)
                model.save_model(str(dest)+'.cbm')
                expected=model.get_best_score()['validation']['AUC'] if not args.iterations else None
                del da,db,frames
            auc=float(roc_auc_score(hf.y[va],vp))
            if expected is not None:assert abs(auc-expected)<1e-6,(auc,expected)
            assert np.isfinite(vp).all() and np.isfinite(tp).all()
            np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
            result=dict(model=name,scheme=args.scheme,fold=f,params=params,iterations=iterations,auc=auc,
                anchor_auc=float(roc_auc_score(hf.y[va],anchor)),
                quarter_blend_auc=float(roc_auc_score(hf.y[va],expit(.75*logit(np.clip(anchor,1e-7,1-1e-7))+.25*logit(np.clip(vp,1e-7,1-1e-7))))),
                seconds=time.time()-start,features=len(names),
                validation_note='Nested cached target features and margins; predictor-only train/test group summaries. Outer-fold early stopping in screening; fixed length when iterations is specified.')
            dest.with_suffix('.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
            del model;gc.collect()
        del data;gc.collect()


if __name__=='__main__':main()
