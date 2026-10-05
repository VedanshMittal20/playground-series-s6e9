"""Validate complete OOF coverage and compare a small fixed blend menu."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scheme',choices=['three','ten'],default='ten')
    ap.add_argument('--models',nargs='+',default=['heuljax_xgb','xgb_unlabeled','xgb_combined'])
    ap.add_argument('--stem',default='v3');args=ap.parse_args()
    out=ROOT/'experiments_v3';out=out/'tenfold' if args.scheme=='ten' else out
    nf=10 if args.scheme=='ten' else 3
    base=pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv');y=base.target.to_numpy()
    assignments=pd.read_csv(out/'folds.csv') if nf==10 else base
    assert assignments.id.equals(base.id) and np.array_equal(assignments.target,y)
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');test=pd.read_csv(ROOT/'data/test.csv')
    assert sample.id.equals(test.id)
    oof={};preds={};summaries={}
    for name in args.models:
        p=np.full(len(base),np.nan);count=np.zeros(len(base),int);ts=[];info=[]
        for f in range(nf):
            z=np.load(out/f'{name}_fold{f}.npz');idx=z['indices']
            np.testing.assert_array_equal(idx,np.flatnonzero(assignments.fold==f))
            assert len(z['test'])==len(test) and np.isfinite(z['validation']).all() and np.isfinite(z['test']).all()
            p[idx]=z['validation'];count[idx]+=1;ts.append(z['test']);info.append(json.loads((out/f'{name}_fold{f}.json').read_text()))
        assert np.all(count==1) and np.isfinite(p).all()
        oof[name]=p;preds[name]=np.mean(ts,axis=0);summaries[name]=info
    candidates={k:{k:1.} for k in args.models}
    candidates['equal_logit']={k:1/len(args.models) for k in args.models}
    if set(args.models)=={'heuljax_xgb','xgb_unlabeled','xgb_combined'}:
        candidates.update(combined_half={'xgb_combined':.5,'heuljax_xgb':.25,'xgb_unlabeled':.25},
                          combined_three_quarters={'xgb_combined':.75,'xgb_unlabeled':.25},
                          combined_unlabeled_equal={'xgb_combined':.5,'xgb_unlabeled':.5})
    if set(args.models)=={'heuljax_xgb','xgb_unlabeled','xgb_combined','xgb_combined_rich'}:
        # A small fixed menu: preserve the previous blend, test the richer view,
        # and average the two closely related combined models as one family.
        candidates.update(
            original_three_equal={'heuljax_xgb':1/3,'xgb_unlabeled':1/3,'xgb_combined':1/3},
            combined_unlabeled_equal={'xgb_combined':.5,'xgb_unlabeled':.5},
            rich_unlabeled_equal={'xgb_combined_rich':.5,'xgb_unlabeled':.5},
            combined_family_unlabeled_equal={'xgb_combined':.25,'xgb_combined_rich':.25,'xgb_unlabeled':.5},
            three_families_equal={'heuljax_xgb':1/3,'xgb_unlabeled':1/3,'xgb_combined':1/6,'xgb_combined_rich':1/6})
    cp={k:expit(sum(w*logit(np.clip(oof[m],1e-7,1-1e-7)) for m,w in weights.items())) for k,weights in candidates.items()}
    scores={k:float(roc_auc_score(y,p)) for k,p in cp.items()}
    selected=max(scores,key=scores.get);weights=candidates[selected];v=cp[selected]
    tp=expit(sum(w*logit(np.clip(preds[m],1e-7,1-1e-7)) for m,w in weights.items()))
    # Keep the simpler model if its result is essentially tied with a blend.
    for name in args.models:
        if scores[name]>=scores[selected]-1e-5:
            selected=name;weights=candidates[name];v=cp[name];tp=preds[name];break
    baseline=float(roc_auc_score(y,base.equal_logit_blend));assert scores[selected]>baseline
    sub=sample.copy();sub.Will_Buy_EV=tp
    assert list(sub)==['id','Will_Buy_EV'] and sub.id.equals(test.id) and sub.id.is_unique
    assert np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
    dest=ROOT/f'submission_{args.stem}.csv'
    if (ROOT/f'submission_result_{args.stem}.json').exists():raise RuntimeError('Do not overwrite an already submitted artifact; use another stem')
    sub.to_csv(dest,index=False)
    def influence(p):
        pos=p[y==1];neg=p[y==0];m=len(pos);n=len(neg);r=rankdata(np.r_[pos,neg])
        return (r[:m]-rankdata(pos))/n,1-(r[m:]-rankdata(neg))/m
    av,an=influence(v);bv,bn=influence(base.equal_logit_blend.to_numpy())
    delta=scores[selected]-baseline;se=float(np.sqrt(np.var(av-bv,ddof=1)/len(av)+np.var(an-bn,ddof=1)/len(an)))
    metrics=dict(scheme=args.scheme,models=args.models,selected=selected,weights=weights,candidate_auc=scores,
        fold_auc={k:[float(roc_auc_score(y[assignments.fold==f],p[assignments.fold==f])) for f in range(nf)] for k,p in cp.items()},
        baseline_auc=baseline,auc_improvement=delta,approx_paired_auc_difference_ci95=[delta-1.96*se,delta+1.96*se],
        limitations='All method/blend selection reuses training labels; not an independent holdout. Paired interval treats predictions as fixed and omits selection and CV training dependence.',
        transductive_features='Predictor-only train/test group moments. No test labels or leaderboard probing.',
        prediction_method='Mean test probabilities over fold models per family; weighted logit blend across families.',
        submission_rows=len(sub),submission_file=dest.name,submission_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        input_sha256=json.loads((ROOT/'metrics.json').read_text())['input_sha256'],fold_models=summaries)
    (ROOT/f'metrics_{args.stem}.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({'id':base.id,'target':y,'fold':assignments.fold,**oof,'selected':v}).to_csv(out/f'oof_{args.stem}.csv',index=False)
    pd.DataFrame({'id':test.id,**preds,'selected':tp}).to_csv(out/f'test_{args.stem}.csv',index=False)
    print(json.dumps({k:metrics[k] for k in ['selected','weights','candidate_auc','auc_improvement','approx_paired_auc_difference_ci95','submission_sha256']},indent=2))


if __name__=='__main__':main()
