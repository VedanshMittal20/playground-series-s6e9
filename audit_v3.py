"""Independently audit fold indices, input hashes and cached target encodings."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v3'/'tenfold'
parser=argparse.ArgumentParser()
parser.add_argument('--require-complete',action='store_true')
args=parser.parse_args()
train=pd.read_csv(ROOT/'data/train.csv')
folds=pd.read_csv(OUT/'folds.csv')
y=train.Will_Buy_EV.map({'No':0,'Yes':1}).to_numpy()
assert train.id.equals(folds.id) and np.array_equal(y,folds.target)
hashes={name:hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
        for name in ['train.csv','test.csv','sample_submission.csv']}
assert hashes==json.loads((ROOT/'metrics.json').read_text())['input_sha256']
checked=[]
for f in range(10):
    # A saved model result establishes that writing the feature cache completed.
    if not (OUT/f'xgb_unlabeled_fold{f}.json').exists():
        assert not args.require_complete, f'Missing completed fold {f}'
        continue
    matrices,names,state=joblib.load(OUT/f'boost_features_fold{f}.joblib',mmap_mode='r')
    tr=np.flatnonzero(folds.fold!=f);va=np.flatnonzero(folds.fold==f)
    assert len(np.intersect1d(tr,va))==0 and len(tr)+len(va)==len(train)
    assert matrices[0].shape[0]==len(tr) and matrices[1].shape[0]==len(va)
    assert 'id' not in names and 'Will_Buy_EV' not in names
    assert np.isclose(state['encoders']['10.0'].target_mean_,y[tr].mean(),rtol=0,atol=1e-12)
    col=names.index('Annual_Income_USD_te10')
    def manual(donor,query):
        d=pd.DataFrame({'income':train.Annual_Income_USD.iloc[donor].to_numpy(),'y':y[donor]})
        stats=d.groupby('income').y.agg(['sum','count'])
        smooth=(stats['sum']+10*y[donor].mean())/(stats['count']+10)
        return train.Annual_Income_USD.iloc[query].map(smooth).fillna(y[donor].mean()).to_numpy()
    sample=np.linspace(0,len(va)-1,100,dtype=int)
    np.testing.assert_allclose(matrices[1][sample,col],manual(tr,va[sample]),rtol=1e-6,atol=1e-7)
    # Reconstruct inner donor partitions independently and verify recipient rows.
    for donor,receiver in StratifiedKFold(5,shuffle=True,random_state=2026).split(tr,y[tr]):
        sample=receiver[np.linspace(0,len(receiver)-1,20,dtype=int)]
        assert not np.intersect1d(tr[donor],tr[sample]).size
        np.testing.assert_allclose(matrices[0][sample,col],manual(tr[donor],tr[sample]),rtol=1e-6,atol=1e-7)
    for name in ['heuljax_xgb','xgb_unlabeled','xgb_combined','xgb_combined_rich']:
        p=OUT/f'{name}_fold{f}.npz'
        if not p.exists():
            assert not args.require_complete, f'Missing predictions {p.name}'
            continue
        z=np.load(p);np.testing.assert_array_equal(z['indices'],va)
        assert len(z['test'])==286571
        for key in ['validation','test']:
            assert np.isfinite(z[key]).all() and np.all((z[key]>=0)&(z[key]<=1))
    checked.append(f)
assert not args.require_complete or checked==list(range(10))
result=dict(status='passed',complete_required=args.require_complete,folds=checked,input_sha256=hashes,
    checks=['input hashes unchanged','IDs/labels aligned','outer partitions disjoint',
            'training-only target prior','manual outer held-out income encoding matches cache',
            'manual inner donor income encoding matches cache','prediction indices and probability bounds'])
(OUT/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
