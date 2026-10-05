"""Finish predictions from a saved additive model without refitting."""
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from improve import ROOT,OUT,additive_predict

start=time.time()
name='additive_long'; fold=0
base=pd.read_csv(ROOT/'outputs/oof_predictions.csv')
train=pd.read_csv(ROOT/'data/train.csv'); test=pd.read_csv(ROOT/'data/test.csv')
x=train.drop(columns=['id','Will_Buy_EV']); xt=test.drop(columns='id')
tr=np.flatnonzero(base.fold.to_numpy()!=fold); va=np.flatnonzero(base.fold.to_numpy()==fold)
a,b,c=x.iloc[tr].copy(),x.iloc[va].copy(),xt.copy()
model=lgb.Booster(model_file=str(OUT/f'{name}_fold{fold}.txt'))
model.params['num_threads']=8
cats=a.select_dtypes(include=['object','string']).columns
for col,values in zip(cats,model.pandas_categorical):
    dtype=pd.CategoricalDtype(values)
    for df in [a,b,c]:
        df[col]=df[col].where(df[col].isin(values)).astype(dtype)
groups=[[i] for i in range(a.shape[1])]
vp,tp=additive_predict(model,a,[b,c],groups)
np.savez_compressed(OUT/f'{name}_fold{fold}.npz',indices=va,validation=vp,test=tp)
result=dict(model=name,fold=fold,auc=float(roc_auc_score(base.target.iloc[va],vp)),baseline_auc=float(roc_auc_score(base.target.iloc[va],base.blend.iloc[va])),prediction_seconds=time.time()-start)
(OUT/f'{name}_fold{fold}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
curve={}
for n in [1000,2000,3500,5000,7000,10000]:
    curve[n]=float(roc_auc_score(base.target.iloc[va],additive_predict(model,a,[b],groups,n)[0]))
    print(n,curve[n],flush=True)
(OUT/f'{name}_curve_fold{fold}.json').write_text(json.dumps(curve,indent=2))
