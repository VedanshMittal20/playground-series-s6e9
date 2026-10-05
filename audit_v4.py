"""Audit all V4 folds and reload saved models for independent test inference."""
import hashlib
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from transductive_v3 import unlabeled_features
import heuljax_features as hf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v4/ten'
base=pd.read_csv(ROOT/'experiments_v3/tenfold/folds.csv')
train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
assert train.id.equals(base.id) and np.array_equal(base.target,hf.y)
hashes={n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()
        for n in ['train.csv','test.csv','sample_submission.csv']}
assert hashes==json.loads((ROOT/'metrics.json').read_text())['input_sha256']
records=[]
for name,trees in [('lgb_combined',900),('lgb_linear',600)]:
    coverage=np.zeros(len(base),int)
    for f in range(10):
        z=np.load(OUT/f'{name}_fold{f}.npz')
        info=json.loads((OUT/f'{name}_fold{f}.json').read_text())
        idx=np.flatnonzero(base.fold==f)
        np.testing.assert_array_equal(z['indices'],idx);coverage[idx]+=1
        assert len(z['test'])==len(test) and len(z['validation'])==len(idx)
        for key in ['test','validation']:
            assert np.isfinite(z[key]).all() and np.all((z[key]>=0)&(z[key]<=1))
        auc=float(roc_auc_score(base.target.iloc[idx],z['validation']))
        assert abs(auc-info['auc'])<1e-12
        assert info['iterations']==trees and info['scheme']=='ten'
        assert info['params']['seed']==2026+f
        model=lgb.Booster(model_file=str(OUT/f'{name}_fold{f}.txt'))
        assert model.num_trees()==trees and model.num_feature()==522
        records.append(dict(model=name,fold=f,trees=trees,auc=auc))
        del model
    assert np.all(coverage==1)

# Rebuild dispersed test rows without reading prediction arrays as features.
_,ut,un=unlabeled_features(train.drop(columns=['id','Will_Buy_EV']),test.drop(columns='id'),rich=True)
ra,rb,rc,am,bm,cm=joblib.load(ROOT/'experiments_v3/tenfold/heuljax_features_fold0.joblib',mmap_mode='r')
matrices,names,_=joblib.load(ROOT/'experiments_v3/tenfold/boost_features_fold0.joblib',mmap_mode='r')
cols=hf.FEATURE_COLS+['additional_'+n for n in names]+un
sample=np.linspace(0,len(test)-1,120,dtype=int)
raw=np.column_stack([rc[sample],matrices[2][sample],ut[sample]])
for name in ['lgb_combined','lgb_linear']:
    model=lgb.Booster(model_file=str(OUT/f'{name}_fold0.txt'))
    assert model.feature_name()==cols
    features=raw.copy()
    if name=='lgb_linear':
        scaler=np.load(OUT/'lgb_linear_fold0_scaler.npz')
        center=np.zeros(len(cols),dtype=features.dtype);spread=np.ones_like(center)
        center[scaler['columns']]=scaler['mean'];spread[scaler['columns']]=scaler['scale']
        features-=center;features/=spread
    prediction=expit(model.predict(features,raw_score=True,num_threads=8)+cm[sample])
    saved=np.load(OUT/f'{name}_fold0.npz')['test'][sample]
    np.testing.assert_allclose(prediction,saved,rtol=1e-6,atol=1e-8)

metrics=json.loads((ROOT/'metrics_v4_ten.json').read_text())
sub=pd.read_csv(ROOT/'submission_v4.csv');sample_sub=pd.read_csv(ROOT/'data/sample_submission.csv')
assert list(sub)==['id','Will_Buy_EV'] and sub.id.equals(test.id) and sub.id.equals(sample_sub.id)
assert sub.id.is_unique and len(sub)==286571
assert np.isfinite(sub.Will_Buy_EV).all() and sub.Will_Buy_EV.between(0,1).all()
assert hashlib.sha256((ROOT/'submission_v4.csv').read_bytes()).hexdigest()==metrics['submission_sha256']
result=dict(status='passed',folds=records,input_sha256=hashes,submission_sha256=metrics['submission_sha256'],
    checks=['unchanged inputs and ID/label alignment','one held-out prediction per row for both learners',
            'all twenty saved model tree counts match frozen training lengths',
            'recorded AUC matches prediction arrays','saved-model test inference agrees on 120 dispersed rows for both learners',
            'initial margins added once and saved linear scaling reapplied','CSV schema, IDs, bounds and hash'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'status':result['status'],'models_checked':len(records),'checks':result['checks']},indent=2))
