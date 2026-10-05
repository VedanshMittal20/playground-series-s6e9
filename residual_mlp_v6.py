"""Small NumPy residual MLP on nested features, with donor-only logit offsets."""
import argparse
import gc
import json
import time
from collections import deque
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent


def initialize(dim,seed):
    rng=np.random.default_rng(seed)
    return dict(w1=(rng.normal(size=(dim,64))*np.sqrt(2/dim)).astype('float32'),b1=np.zeros(64,'float32'),
        w2=(rng.normal(size=(64,32))*np.sqrt(2/64)).astype('float32'),b2=np.zeros(32,'float32'),
        w3=np.zeros(32,'float32'),b3=np.zeros(1,'float32'),skip=np.zeros(dim,'float32'))


def forward(x,margin,p,rng=None):
    z1=x@p['w1']+p['b1'];s1=expit(z1);h1=z1*s1
    mask1=1. if rng is None else (rng.random(h1.shape)<.95).astype('float32')/.95
    h1=h1*mask1
    z2=h1@p['w2']+p['b2'];s2=expit(z2);h2=z2*s2
    mask2=1. if rng is None else (rng.random(h2.shape)<.95).astype('float32')/.95
    h2=h2*mask2
    z=margin+x@p['skip']+h2@p['w3']+p['b3'][0]
    return z,(z1,s1,h1,mask1,z2,s2,h2,mask2)


def gradients(x,margin,y,p,rng=None):
    z,c=forward(x,margin,p,rng)
    z1,s1,h1,m1,z2,s2,h2,m2=c
    d=(expit(z)-y)/len(y)
    g=dict(skip=x.T@d,w3=h2.T@d,b3=np.array([d.sum()],dtype='float32'))
    d2=d[:,None]*p['w3']*m2*(s2+z2*s2*(1-s2))
    g.update(w2=h1.T@d2,b2=d2.sum(axis=0))
    d1=(d2@p['w2'].T)*m1*(s1+z1*s1*(1-s1))
    g.update(w1=x.T@d1,b1=d1.sum(axis=0))
    return g,float(np.mean(np.logaddexp(0,z)-y*z))


def infer(x,margin,p):
    result=np.empty(len(x),dtype='float32')
    for start in range(0,len(x),8192):
        stop=min(start+8192,len(x));result[start:stop]=expit(forward(x[start:stop],margin[start:stop],p)[0])
    return result


def fit(x,margin,y,v,vm,vy,epochs,seed,fixed=False,average_last=1):
    p=initialize(x.shape[1],seed);rng=np.random.default_rng(seed)
    first={k:np.zeros_like(a) for k,a in p.items()};second={k:np.zeros_like(a) for k,a in p.items()}
    best=-np.inf;best_p=None;best_epoch=0;step=0;history=[];snapshots=deque(maxlen=average_last)
    for epoch in range(1,epochs+1):
        start=time.time();order=rng.permutation(len(y));loss=[]
        # Keep the same schedule when a screened epoch count is frozen later.
        lr=.00015+.00085*.5*(1+np.cos(np.pi*min(epoch-1,35)/35))
        for begin in range(0,len(y),4096):
            idx=order[begin:begin+4096];g,batch_loss=gradients(x[idx],margin[idx],y[idx],p,rng)
            loss.append(batch_loss);step+=1
            for k in ['w1','w2','w3','skip']:g[k]+=1e-4*p[k]
            norm=float(np.sqrt(sum(np.sum(a*a) for a in g.values())))
            multiplier=min(1.,2./max(norm,1e-12))
            for k in p:
                grad=g[k]*multiplier;first[k]*=.9;first[k]+=.1*grad
                second[k]*=.999;second[k]+=.001*grad*grad
                p[k]-=lr*(first[k]/(1-.9**step))/(np.sqrt(second[k]/(1-.999**step))+1e-8)
        snapshots.append({k:a.copy() for k,a in p.items()})
        evaluated=p if average_last==1 else {k:np.mean(np.stack([s[k] for s in snapshots]),axis=0) for k in p}
        pred=infer(v,vm,evaluated);auc=float(roc_auc_score(vy,pred))
        row=dict(epoch=epoch,auc=auc,validation_logloss=float(log_loss(vy,pred)),training_loss=float(np.mean(loss)),seconds=time.time()-start)
        history.append(row);print(json.dumps(row),flush=True)
        if auc>best+1e-7:best=auc;best_epoch=epoch;best_p={k:a.copy() for k,a in evaluated.items()}
        if not fixed and epoch-best_epoch>=7:break
    return (evaluated,epoch,history) if fixed else (best_p,best_epoch,history)


def prepare(cache,tr,va,train,test,rich):
    import heuljax_features as hf
    from transductive_v3 import unlabeled_features
    a,b,c,am,bm,cm=joblib.load(cache,mmap_mode='r')
    numeric=[i for i,t in enumerate(hf.FEATURE_TYPES) if t=='q']
    categorical=[i for i,t in enumerate(hf.FEATURE_TYPES) if t=='c']
    names=[hf.FEATURE_COLS[i] for i in numeric]
    # Extra logit coordinates make probability features easier to combine.
    rate=[i for i in numeric if hf.FEATURE_COLS[i] in hf.RATE_COLS]
    names+=['logit_'+hf.FEATURE_COLS[i] for i in rate]
    categories={i:np.unique(a[:,i]) for i in categorical}
    names += [f'{hf.FEATURE_COLS[i]}={float(value)}' for i,values in categories.items() for value in values]
    ux,ut,un=unlabeled_features(train.drop(columns=['id','Will_Buy_EV']),test.drop(columns='id'),rich=rich)
    names+=un
    data=[]
    for raw,u in zip([a,b,c],[ux[tr],ux[va],ut]):
        parts=[raw[:,numeric],logit(np.clip(raw[:,rate],1e-5,1-1e-5))]
        parts += [(raw[:,i,None]==values[None,:]).astype('float32') for i,values in categories.items()]
        parts.append(u);data.append(np.column_stack(parts).astype('float32',copy=False))
    del a,b,c,ux,ut;gc.collect()
    center=np.empty(len(names),'float32');spread=np.empty_like(center)
    for begin in range(0,len(names),32):
        end=min(begin+32,len(names));block=data[0][:,begin:end]
        center[begin:end]=np.nanmean(block,axis=0);spread[begin:end]=np.nanstd(block,axis=0)
    center[~np.isfinite(center)]=0;spread[(~np.isfinite(spread))|(spread<1e-4)]=1
    for d in data:
        d-=center;d/=spread;np.nan_to_num(d,copy=False,nan=0.,posinf=6.,neginf=-6.);np.clip(d,-6,6,out=d)
    return data,[np.asarray(m,dtype='float32') for m in [am,bm,cm]],dict(names=names,center=center,spread=spread,categories=categories)


def check_gradient():
    rng=np.random.default_rng(5);x=rng.normal(size=(7,4)).astype('float32');margin=rng.normal(size=7).astype('float32');y=(rng.random(7)>.5).astype('float32')
    p=initialize(4,5);p['w3'][:]=rng.normal(size=32)*.1;p['skip'][:]=rng.normal(size=4)*.1
    grad,_=gradients(x,margin,y,p)
    for k in p:
        flat=p[k].reshape(-1)
        for idx in sorted(set([0,len(flat)//2,len(flat)-1])):
            original=flat[idx];eps=.002
            flat[idx]=original+eps;_,plus=gradients(x,margin,y,p)
            flat[idx]=original-eps;_,minus=gradients(x,margin,y,p)
            flat[idx]=original
            np.testing.assert_allclose(grad[k].reshape(-1)[idx],(plus-minus)/(2*eps),rtol=.015,atol=3e-5)
    print('Gradient check passed for every parameter group.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folds',nargs='+',type=int,default=[0]);ap.add_argument('--scheme',choices=['three','ten'],default='three')
    ap.add_argument('--epochs',type=int,default=35);ap.add_argument('--fixed',action='store_true');ap.add_argument('--check-gradient',action='store_true')
    ap.add_argument('--name',choices=['residual_mlp','residual_mlp_average'],default='residual_mlp')
    ap.add_argument('--average-last',type=int,default=1);args=ap.parse_args()
    assert args.average_last>=1
    if args.name=='residual_mlp' and args.average_last!=1:raise ValueError('Use the separate average variant name')
    if args.check_gradient:check_gradient();return
    import heuljax_features as hf
    source=ROOT/'experiments_v3';source=source/'tenfold' if args.scheme=='ten' else source
    out=ROOT/'experiments_v6'/args.scheme;out.mkdir(parents=True,exist_ok=True)
    base=pd.read_csv(source/'folds.csv' if args.scheme=='ten' else ROOT/'experiments_v2/oof_predictions.csv')
    train=pd.read_csv(ROOT/'data/train.csv');test=pd.read_csv(ROOT/'data/test.csv')
    assert train.id.equals(base.id) and np.array_equal(base.target,hf.y)
    for f in args.folds:
        dest=out/f'{args.name}_fold{f}'
        if dest.with_suffix('.json').exists() and dest.with_suffix('.npz').exists():continue
        tr=np.flatnonzero(base.fold!=f);va=np.flatnonzero(base.fold==f);start=time.time()
        with threadpool_limits(limits=8):
            data,margins,state=prepare(source/f'heuljax_features_fold{f}.joblib',tr,va,train,test,rich=False)
            print(f'Prepared fold {f}: {data[0].shape[1]} features',flush=True)
            p,epochs,history=fit(data[0],margins[0],hf.y[tr].astype('float32'),data[1],margins[1],hf.y[va],args.epochs,2026+f,fixed=args.fixed,average_last=args.average_last)
            vp=infer(data[1],margins[1],p);tp=infer(data[2],margins[2],p)
            np.savez_compressed(dest.with_suffix('.npz'),indices=va,validation=vp,test=tp)
            joblib.dump(dict(parameters=p,preprocessing=state),dest.with_name(dest.name+'_model').with_suffix('.joblib'))
        rich=np.load(source/f'xgb_combined_rich_fold{f}.npz')['validation'];unlabeled=np.load(source/f'xgb_unlabeled_fold{f}.npz')['validation']
        anchor=(logit(np.clip(rich,1e-7,1-1e-7))+logit(np.clip(unlabeled,1e-7,1-1e-7)))/2
        result=dict(model=args.name,scheme=args.scheme,fold=f,epochs=epochs,max_epochs=args.epochs,fixed=args.fixed,
            auc=float(roc_auc_score(hf.y[va],vp)),anchor_auc=float(roc_auc_score(hf.y[va],anchor)),
            quarter_blend_auc=float(roc_auc_score(hf.y[va],.75*anchor+.25*logit(np.clip(vp,1e-7,1-1e-7)))),
            history=history,seconds=time.time()-start,features=len(state['names']),
            params=dict(hidden=[64,32],batch_size=4096,dropout=.05,weight_decay=1e-4,seed=2026+f,
                        learning_rate_initial=.001,learning_rate_minimum=.00015,cosine_schedule_epochs=35),
            validation_note='Donor-only features and offsets. Scaling and categorical vocabulary fitted on outer training rows. Outer AUC checkpoint selection in screening, fixed final epoch when --fixed.')
        result['params']['average_last_epochs']=args.average_last
        dest.with_suffix('.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
        del data,margins,p,state;gc.collect()


if __name__=='__main__':main()
