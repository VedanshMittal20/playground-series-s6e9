"""Test unlabeled income-group covariates as a supplementary feature view.

Only predictor columns from train/test enter the group aggregates. All target
encodings come from the previously cached inner-cross-fitted donor features.
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v3'


def unlabeled_features(x,xt,rich=False):
    joint=pd.concat([x,xt],ignore_index=True)
    values=pd.get_dummies(joint.drop(columns='Annual_Income_USD'),dtype=float)
    values['gate_score']=.6*joint.Environmental_Concern_Level+2*joint.Subsidy_Available.eq('Yes')-joint.Range_Anxiety_Level.map({'Low':0,'Medium':1,'High':3})
    for threshold in [0,1,2,3,4,5]: values[f'gate_above{threshold}']=(values.gate_score>=threshold).astype(float)
    if rich:
        gate=(joint.Environmental_Concern_Level.astype(int)-1)*6+joint.Subsidy_Available.eq('Yes')*3+joint.Range_Anxiety_Level.map({'Low':0,'Medium':1,'High':2})
        for level in range(30):values[f'gate_joint{level}']=(gate==level).astype(float)
        for threshold in [2.,3.,4.,5.]:
            for slope in [1.,2.,4.]:values[f'gate_soft{threshold}_{slope}']=expit(slope*(values.gate_score-threshold))
    pieces=[]; names=[]
    for width in [1,100,1000]:
        group=np.floor(joint.Annual_Income_USD/width).astype(int)
        averages=values.groupby(group).transform('mean')
        pieces.append(averages.to_numpy(np.float32)); names.extend([f'unlabeled_inc{width}_{c}' for c in values])
        pieces.append(group.map(group.value_counts()).to_numpy(np.float32)[:,None]); names.append(f'unlabeled_inc{width}_count')
    result=np.column_stack(pieces)
    return result[:len(x)],result[len(x):],names


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',nargs='+',type=int,default=[0]);args=ap.parse_args()
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    base=pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv');y=train.Will_Buy_EV.map({'No':0,'Yes':1}).to_numpy()
    assert train.id.equals(base.id) and np.array_equal(y,base.target)
    ux,ut,un=unlabeled_features(train.drop(columns=['id','Will_Buy_EV']),test.drop(columns='id'))
    for fold in args.folds:
        dest=OUT/f'xgb_unlabeled_fold{fold}'
        if dest.with_suffix('.npz').exists():continue
        start=time.time();tr=np.flatnonzero(base.fold!=fold);va=np.flatnonzero(base.fold==fold)
        matrices,names,_=joblib.load(OUT/f'boost_features_fold{fold}.joblib')
        a,b,c=[np.column_stack([d,u]) for d,u in zip(matrices,[ux[tr],ux[va],ut])];names=names+un
        params=dict(objective='binary:logistic',eval_metric='auc',device='cuda',tree_method='hist',max_bin=512,max_depth=4,eta=.025,reg_alpha=3.,reg_lambda=10.,gamma=.1,min_child_weight=5,subsample=.9,colsample_bytree=.85,nthread=8,seed=2026)
        da=xgb.QuantileDMatrix(a,label=y[tr],max_bin=512,nthread=8,feature_names=names)
        db=xgb.QuantileDMatrix(b,label=y[va],max_bin=512,nthread=8,feature_names=names,ref=da)
        dc=xgb.QuantileDMatrix(c,max_bin=512,nthread=8,feature_names=names,ref=da)
        model=xgb.train(params,da,num_boost_round=5000,evals=[(db,'validation')],early_stopping_rounds=450,verbose_eval=500)
        trees=model.best_iteration+1;vp=model.predict(db,iteration_range=(0,trees));tp=model.predict(dc,iteration_range=(0,trees))
        model.save_model(str(dest)+'.ubj');np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
        result=dict(model='xgb_unlabeled',fold=fold,params=params,iterations=trees,auc=float(roc_auc_score(y[va],vp)),blend_v2_auc=float(roc_auc_score(y[va],expit((logit(vp)+logit(base.equal_logit_blend.iloc[va]))/2))),seconds=time.time()-start,validation_note='Transductive predictor-only aggregates include validation/test features, never their labels; target encodings remain nested cross-fitted; outer-fold early stopping.')
        dest.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
