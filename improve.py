"""Resumable, leakage-safe experiments on the baseline's saved folds."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy import sparse
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments_v2'
OUT.mkdir(exist_ok=True)
NUM = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
       'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work', 'Environmental_Concern_Level']


def additive_predict(model, reference, frames, groups, iterations=None):
    """Evaluate disjoint additive groups once per distinct value combination."""
    anchor=reference.iloc[[0]].copy()
    offset=float(model.predict(anchor,raw_score=True,num_iteration=iterations)[0])
    raw=[np.full(len(df),offset) for df in frames]
    for group in groups:
        cols=[reference.columns[i] for i in group]
        values=pd.concat([df[cols] for df in frames],ignore_index=True).drop_duplicates().reset_index(drop=True)
        rows=pd.concat([anchor]*len(values),ignore_index=True)
        for col in cols:
            rows[col]=values[col]
        values['_effect']=model.predict(rows,raw_score=True,num_iteration=iterations)-offset
        for i,df in enumerate(frames):
            effect=df[cols].merge(values,on=cols,how='left',sort=False,validate='many_to_one')['_effect'].to_numpy()
            assert len(effect)==len(df) and np.isfinite(effect).all()
            raw[i]+=effect
    predictions=[expit(v) for v in raw]
    for df,p in zip(frames,predictions):
        # Check exact equivalence to ordinary tree inference on dispersed rows.
        idx=np.linspace(0,len(df)-1,min(100,len(df)),dtype=int)
        direct=(model.predict(df.iloc[idx],num_iteration=iterations) if isinstance(model,lgb.Booster)
                else model.predict_proba(df.iloc[idx],num_iteration=iterations)[:,1])
        assert np.allclose(direct,p[idx],rtol=1e-8,atol=1e-10)
    return predictions


def features(x):
    x = x.copy()
    for c in ['Age', 'Annual_Income_USD', 'Daily_Commute_km']:
        x[c+'_exact'] = x[c].astype(str)
    for a,b in [('Subsidy_Available','Environmental_Concern_Level'),
                ('Subsidy_Available','Range_Anxiety_Level'),
                ('Environmental_Concern_Level','Range_Anxiety_Level')]:
        x[a+'__'+b] = x[a].astype(str)+'|'+x[b].astype(str)
    return x


def run(name, fold, x, xt, y, baseline):
    dest = OUT / f'{name}_fold{fold}.npz'
    if dest.exists():
        print(f'Using saved {name} fold {fold}', flush=True)
        return
    start=time.time()
    tr=np.flatnonzero(baseline.fold.to_numpy()!=fold)
    va=np.flatnonzero(baseline.fold.to_numpy()==fold)
    a,b,c=(features(d) for d in (x.iloc[tr],x.iloc[va],xt))
    if name.startswith('additive'):
        a,b,c=(d.copy() for d in (x.iloc[tr],x.iloc[va],xt))
        if name=='additive_digits':
            for df in [a,b,c]:
                income=df.Annual_Income_USD.astype('int64')
                for power in range(6):
                    df[f'income_digit_{power}']=(income//(10**power))%10
                for power in [2,3,4]:
                    df[f'income_suffix_{power}']=income%(10**power)
        if name=='additive_original':
            original=pd.read_csv(ROOT/'original/EV_Adoption_and_Range_Anxiety_Dataset.csv')
            original['_y']=original.Will_Buy_EV.map({'No':0,'Yes':1})
            for col in ['Annual_Income_USD','Daily_Commute_km','Age']:
                stats=original.groupby(col)._y.agg(['sum','count'])
                rate=(stats['sum']+5*original._y.mean())/(stats['count']+5)
                for df in [a,b,c]:
                    df[col+'_source_rate']=df[col].map(rate).fillna(original._y.mean())
                    df[col+'_source_count']=df[col].map(stats['count']).fillna(0)
    print(f'Start {name} fold {fold}',flush=True)
    if name.startswith('logistic'):
        # Exact values are regularized categories; splines provide smooth fallback.
        enc=OneHotEncoder(handle_unknown='ignore',dtype=np.float64)
        aa=enc.fit_transform(a.astype(str)); bb=enc.transform(b.astype(str)); cc=enc.transform(c.astype(str))
        spl=SplineTransformer(n_knots=12,degree=3,knots='quantile',extrapolation='linear')
        sa=spl.fit_transform(a[NUM]); sb=spl.transform(b[NUM]); sc=spl.transform(c[NUM])
        mats=[]
        for df,oh,sp in [(a,aa,sa),(b,bb,sb),(c,cc,sc)]:
            subsidy=df.Subsidy_Available.eq('Yes').to_numpy()[:,None]
            mats.append(sparse.hstack([oh,sp,sp*subsidy],format='csr'))
        model=LogisticRegression(C=0.3 if name=='logistic' else 3.0,solver='lbfgs',max_iter=500,tol=1e-5)
        with threadpool_limits(limits=8):
            model.fit(mats[0],y.iloc[tr])
            vp=model.predict_proba(mats[1])[:,1]; tp=model.predict_proba(mats[2])[:,1]
        joblib.dump(dict(model=model,encoder=enc,splines=spl),OUT/f'{name}_fold{fold}.joblib')
    else:
        cats=a.select_dtypes(include=['object','string']).columns.tolist()
        if name=='target_encoded':
            # Inner cross-fitting protects training rows; outer labels are never used.
            enc=TargetEncoder(smooth=20,cv=5,shuffle=True,random_state=2026,target_type='binary')
            aa=enc.fit_transform(a.astype(str),y.iloc[tr]); bb=enc.transform(b.astype(str)); cc=enc.transform(c.astype(str))
            for i,col in enumerate(a.columns.tolist()):
                a[col+'_te']=aa[:,i]; b[col+'_te']=bb[:,i]; c[col+'_te']=cc[:,i]
            joblib.dump(enc,OUT/f'{name}_encoder_fold{fold}.joblib')
        if name.startswith('cat_'):
            cat_extra=dict(task_type='GPU',devices='0',gpu_ram_part=.5,max_ctr_complexity=1,border_count=254) if name.startswith('cat_gpu') else {}
            model=CatBoostClassifier(iterations=3000 if name=='cat_gpu_shallow' else (1800 if name=='cat_gpu' else 1400),depth=4 if name=='cat_gpu_shallow' else 6,learning_rate=.06,l2_leaf_reg=5,
                loss_function='Logloss',random_seed=2026,thread_count=8,verbose=350,allow_writing_files=False)
            model.set_params(**cat_extra)
            model.fit(a,y.iloc[tr],cat_features=cats)
            model.save_model(str(OUT/f'{name}_fold{fold}.cbm'))
        else:
            for col in cats:
                dtype=pd.CategoricalDtype(sorted(a[col].unique()))
                a[col],b[col],c[col]=(d[col].astype(dtype) for d in (a,b,c))
            extra={}
            if name.startswith('additive'):
                extra=dict(max_bin=16383,min_data_in_bin=1,
                           interaction_constraints=[[i] for i in range(a.shape[1])])
                if name=='additive_fine':
                    extra['bin_construct_sample_cnt']=1000000
                if name=='additive_digits':
                    income_group=[i for i,col in enumerate(a.columns) if col=='Annual_Income_USD' or col.startswith('income_')]
                    extra['interaction_constraints']=[income_group]+[[i] for i in range(a.shape[1]) if i not in income_group]
            trees=5000 if name=='additive_fine' else (7000 if name=='additive_7000' else (6000 if name=='additive_digits' else (10000 if name=='additive_long' else (3500 if extra else 1200))))
            model=lgb.LGBMClassifier(n_estimators=trees,learning_rate=.05 if extra else .035,num_leaves=31 if name=='additive_fine' else 15,
                min_child_samples=20 if name=='additive_fine' else 150,reg_lambda=5 if name=='additive_fine' else 10,colsample_bytree=.95,
                cat_smooth=20,cat_l2=10,max_cat_threshold=64,
                random_state=2026,n_jobs=8,verbosity=-1,**extra)
            model.fit(a,y.iloc[tr],categorical_feature=cats)
            model.booster_.save_model(str(OUT/f'{name}_fold{fold}.txt'))
        if name.startswith('additive'):
            vp,tp=additive_predict(model,a,[b,c],extra['interaction_constraints'])
        else:
            vp=model.predict_proba(b)[:,1]; tp=model.predict_proba(c)[:,1]
        if name=='additive_long':
            curve={n:float(roc_auc_score(y.iloc[va],additive_predict(model,a,[b],extra['interaction_constraints'],n)[0])) for n in [1000,2000,3500,5000,7000,10000]}
            (OUT/f'{name}_curve_fold{fold}.json').write_text(json.dumps(curve,indent=2))
            print('Learning curve '+json.dumps(curve),flush=True)
    assert np.isfinite(vp).all() and np.isfinite(tp).all()
    np.savez_compressed(dest,indices=va,validation=vp,test=tp)
    result=dict(model=name,fold=fold,auc=float(roc_auc_score(y.iloc[va],vp)),
        baseline_auc=float(roc_auc_score(y.iloc[va],baseline.blend.iloc[va])),seconds=time.time()-start)
    (OUT/f'{name}_fold{fold}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',default=['lgb_exact','target_encoded','logistic','cat_exact'])
    parser.add_argument('--folds',nargs='+',type=int,default=[0])
    args=parser.parse_args()
    train=pd.read_csv(ROOT/'data/train.csv'); test=pd.read_csv(ROOT/'data/test.csv')
    baseline=pd.read_csv(ROOT/'outputs/oof_predictions.csv')
    assert train.id.equals(baseline.id)
    y=train.Will_Buy_EV.map({'No':0,'Yes':1})
    assert np.array_equal(y,baseline.target)
    x=train.drop(columns=['id','Will_Buy_EV']); xt=test.drop(columns='id')
    for name in args.models:
        for fold in args.folds:
            run(name,fold,x,xt,y,baseline)
