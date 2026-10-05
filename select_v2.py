"""Compare completed experiments and write one locally selected submission."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments_v2'
parser=argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',required=True)
args=parser.parse_args()
base=pd.read_csv(ROOT/'outputs/oof_predictions.csv')
test=pd.read_csv(ROOT/'data/test.csv')
sample=pd.read_csv(ROOT/'data/sample_submission.csv')
assert sample.id.equals(test.id)
y=base.target.to_numpy()
oof={}; preds={}
for name in args.models:
    p=np.full(len(base),np.nan); counts=np.zeros(len(base),dtype=int); t=[]
    for fold in range(3):
        z=np.load(OUT/f'{name}_fold{fold}.npz')
        idx=z['indices']
        assert np.array_equal(idx,np.flatnonzero(base.fold.to_numpy()==fold))
        assert len(z['test'])==len(test)
        p[idx]=z['validation']; counts[idx]+=1; t.append(z['test'])
    assert (counts==1).all() and np.isfinite(p).all()
    oof[name]=p; preds[name]=np.mean(t,axis=0)
if len(args.models)==2:
    for kind in ['probability','logit']:
        key=f'equal_{kind}_blend'
        transform=(lambda v: logit(np.clip(v,1e-7,1-1e-7))) if kind=='logit' else (lambda v:v)
        inverse=expit if kind=='logit' else (lambda v:v)
        oof[key]=inverse(np.mean([transform(oof[k]) for k in args.models],axis=0))
        preds[key]=inverse(np.mean([transform(preds[k]) for k in args.models],axis=0))
scores={k:float(roc_auc_score(y,v)) for k,v in oof.items()}
selected=max(scores,key=scores.get)
baseline_auc=float(roc_auc_score(y,base.blend))
assert scores[selected]>baseline_auc
fold_scores={k:[float(roc_auc_score(y[base.fold==f],v[base.fold==f])) for f in range(3)] for k,v in oof.items()}
baseline_folds=[float(roc_auc_score(y[base.fold==f],base.blend[base.fold==f])) for f in range(3)]
sub=sample.copy(); sub.Will_Buy_EV=preds[selected]
assert list(sub.columns)==['id','Will_Buy_EV'] and sub.id.equals(test.id)
assert sub.id.is_unique and np.isfinite(sub.Will_Buy_EV).all() and sub.Will_Buy_EV.between(0,1).all()
sub.to_csv(ROOT/'submission_v2.csv',index=False)
pd.DataFrame({'id':base.id,'target':y,'fold':base.fold,**oof}).to_csv(OUT/'oof_predictions.csv',index=False)
pd.DataFrame({'id':test.id,**preds}).to_csv(OUT/'test_predictions.csv',index=False)
metrics=dict(selected=selected,source_models=args.models,oof_auc=scores,fold_auc=fold_scores,baseline_auc=baseline_auc,
    baseline_fold_auc=baseline_folds,improvement=scores[selected]-baseline_auc,
    limitations='Fold 0 was used for screening. All folds were used for final model/blend selection; this is not an independent final holdout.',
    test_prediction_method='Equal average over three fold models; chosen blend if applicable',
    submission_rows=len(sub),submission_sha256=hashlib.sha256((ROOT/'submission_v2.csv').read_bytes()).hexdigest())
(ROOT/'metrics_v2.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2))
