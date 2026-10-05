"""Compare a small fixed diversity-blend menu against the preserved XGB anchor."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--scheme',choices=['three','ten'],default='three')
    ap.add_argument('--models',nargs='+',default=['lgb_combined','lgb_linear'])
    ap.add_argument('--stem',default='v4');ap.add_argument('--write-submission',action='store_true')
    args=ap.parse_args();nf=3 if args.scheme=='three' else 10
    out=ROOT/'experiments_v4'/args.scheme
    base=pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv');y=base.target.to_numpy()
    sample=pd.read_csv(ROOT/'data/sample_submission.csv');test_ids=pd.read_csv(ROOT/'data/test.csv',usecols=['id'])
    assert sample.id.equals(test_ids.id)
    if nf==10:
        anchor=pd.read_csv(ROOT/'experiments_v3/tenfold/oof_v3.csv')
        assert anchor.id.equals(base.id) and np.array_equal(anchor.target,y)
        folds=anchor.fold.to_numpy();oof={'anchor':anchor.selected.to_numpy()}
        saved=pd.read_csv(ROOT/'submission_v3.csv');assert saved.id.equals(sample.id)
        tp={'anchor':saved.Will_Buy_EV.to_numpy()}
    else:
        folds=base.fold.to_numpy();oof={};tp={}
        for name in ['xgb_combined_rich','xgb_unlabeled']:
            p=np.full(len(base),np.nan);ps=[]
            for f in range(3):
                z=np.load(ROOT/'experiments_v3'/f'{name}_fold{f}.npz')
                np.testing.assert_array_equal(z['indices'],np.flatnonzero(folds==f))
                p[z['indices']]=z['validation'];ps.append(z['test'])
            oof[name]=p;tp[name]=np.mean(ps,axis=0)
        oof={'anchor':expit(sum(logit(np.clip(p,1e-7,1-1e-7)) for p in oof.values())/2)}
        tp={'anchor':expit(sum(logit(np.clip(p,1e-7,1-1e-7)) for p in tp.values())/2)}
    for name in args.models:
        p=np.full(len(base),np.nan);seen=np.zeros(len(base),int);ps=[]
        for f in range(nf):
            z=np.load(out/f'{name}_fold{f}.npz')
            np.testing.assert_array_equal(z['indices'],np.flatnonzero(folds==f))
            p[z['indices']]=z['validation'];seen[z['indices']]+=1;ps.append(z['test'])
        assert np.all(seen==1)
        oof[name]=p;tp[name]=np.mean(ps,axis=0)
    for d in [oof,tp]:
        for p in d.values():assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    menu={'anchor':{'anchor':1.}}
    for name in args.models:
        menu[name]={name:1.}
        menu[f'{name}_quarter']={'anchor':.75,name:.25}
        menu[f'{name}_half']={'anchor':.5,name:.5}
    if len(args.models)>1:
        menu['diversity_quarter']={'anchor':.75,**{m:.25/len(args.models) for m in args.models}}
        menu['diversity_half']={'anchor':.5,**{m:.5/len(args.models) for m in args.models}}
    def blend(weights,pred):return expit(sum(w*logit(np.clip(pred[n],1e-7,1-1e-7)) for n,w in weights.items()))
    candidates={n:blend(w,oof) for n,w in menu.items()}
    scores={n:float(roc_auc_score(y,p)) for n,p in candidates.items()}
    selected=max(scores,key=scores.get)
    if scores[selected]-scores['anchor']<1e-5:selected='anchor'
    v=candidates[selected];weights=menu[selected]
    def influence(p):
        pos=p[y==1];neg=p[y==0];m=len(pos);n=len(neg);r=rankdata(np.r_[pos,neg])
        return (r[:m]-rankdata(pos))/n,1-(r[m:]-rankdata(neg))/m
    av,an=influence(v);bv,bn=influence(oof['anchor'])
    delta=scores[selected]-scores['anchor'];se=float(np.sqrt(np.var(av-bv,ddof=1)/len(av)+np.var(an-bn,ddof=1)/len(an)))
    result=dict(scheme=args.scheme,selected=selected,weights=weights,candidate_auc=scores,
        fold_auc={n:[float(roc_auc_score(y[folds==f],p[folds==f])) for f in range(nf)] for n,p in candidates.items()},
        auc_improvement=delta,approx_paired_difference_ci95=[delta-1.96*se,delta+1.96*se],
        limitations='Method and blend selection reuse labels; approximate fixed-prediction interval omits this selection and CV training dependence.',
        prediction_method='Mean test probabilities across folds for each new learner, then a weighted logit blend with the preserved XGB anchor.',
        input_sha256=json.loads((ROOT/'metrics.json').read_text())['input_sha256'],
        fold_models={n:[json.loads((out/f'{n}_fold{f}.json').read_text()) for f in range(nf)] for n in args.models})
    if args.write_submission:
        assert args.scheme=='ten' and selected!='anchor'
        if (ROOT/f'submission_result_{args.stem}.json').exists():raise RuntimeError('Submitted artifact is immutable')
        sub=sample.copy();sub.Will_Buy_EV=blend(weights,tp)
        assert sub.id.equals(test_ids.id) and sub.id.is_unique and len(sub)==286571
        dest=ROOT/f'submission_{args.stem}.csv';sub.to_csv(dest,index=False)
        readback=pd.read_csv(dest);assert readback.id.equals(test_ids.id) and readback.Will_Buy_EV.between(0,1).all()
        result.update(submission_file=dest.name,submission_rows=len(sub),submission_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        pd.DataFrame({'id':base.id,'target':y,'fold':folds,**oof,'selected':v}).to_csv(out/f'oof_{args.stem}.csv',index=False)
        pd.DataFrame({'id':sample.id,**tp,'selected':sub.Will_Buy_EV}).to_csv(out/f'test_{args.stem}.csv',index=False)
    (ROOT/f'metrics_{args.stem}_{args.scheme}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['fold_auc','fold_models','input_sha256']},indent=2))


if __name__=='__main__':main()
