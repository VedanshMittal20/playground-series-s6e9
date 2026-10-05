"""Nested left/right income-bin statistics, without recipient-label access.

Motivated by asymmetric-bin features in jazivxt/single-model-zoom-zoom.
This implementation uses fixed-width bins, half-bin offsets, and donor-only
statistics inside each outer fold. No public predictions are used.
"""
import numpy as np
from sklearn.model_selection import KFold


def encode_asymmetric(train_income, valid_income, test_income, y, *, seed=2026):
    income=[np.asarray(v,dtype=np.float64) for v in [train_income,valid_income,test_income]]
    y=np.asarray(y,dtype=np.float64)
    assert len(income[0])==len(y) and all(np.isfinite(v).all() for v in income)
    assert all((v>=0).all() for v in income)
    splits=list(KFold(5,shuffle=True,random_state=seed).split(y))
    parts=[[],[],[]];names=[]
    for width in [8,16,32,64]:
        for offset in [0,width//2]:
            codes=[np.floor((v+offset)/width).astype(np.int64)+1 for v in income]
            size=max(int(c.max(initial=0)) for c in codes)+2
            def table(indices):
                yy=y[indices];cc=codes[0][indices];prior=yy.mean()
                count=np.bincount(cc,minlength=size).astype(np.float64)
                total=np.bincount(cc,weights=yy,minlength=size)
                center=(total+10*prior)/(count+10)
                left=np.r_[prior,center[:-1]];right=np.r_[center[1:],prior]
                return np.column_stack([center,left,right,right-left,
                    center-.5*(left+right),np.log1p(count),
                    np.log1p(np.r_[0,count[:-1]])-np.log1p(np.r_[count[1:],0])]).astype(np.float32)
            inner=np.empty((len(y),7),dtype=np.float32)
            for donors,receivers in splits:
                assert not np.intersect1d(donors,receivers).size
                inner[receivers]=table(donors)[codes[0][receivers]]
            full=table(np.arange(len(y)))
            parts[0].append(inner)
            for i in [1,2]:parts[i].append(full[codes[i]])
            names += [f'asym_income_w{width}_o{offset}_{s}' for s in
                      ['center','left','right','slope','curvature','log_count','support_imbalance']]
    matrices=[np.column_stack(p) for p in parts]
    assert all(np.isfinite(m).all() for m in matrices)
    return matrices,names


def check_recipient_exclusion():
    rng=np.random.default_rng(91)
    income=rng.integers(30000,30200,2000);y=rng.integers(0,2,2000)
    a,names=encode_asymmetric(income,np.array([30001,30150]),np.array([29999,30250]),y)
    changed=y.copy();changed[17]=1-changed[17]
    b,_=encode_asymmetric(income,np.array([30001,30150]),np.array([29999,30250]),changed)
    # Flipping a recipient's label must not alter that recipient's features.
    np.testing.assert_array_equal(a[0][17],b[0][17])
    assert a[0].shape==(2000,56) and len(names)==56
    # Check a concrete held-out central bin against an independent formula.
    codes=income//8+1;query=30001//8+1;mask=codes==query
    expected=(y[mask].sum()+10*y.mean())/(mask.sum()+10)
    np.testing.assert_allclose(a[1][0,0],expected,rtol=1e-6)
    print('Passed: own-label exclusion, held-out bin formula, finite boundary cases, feature shape.')


if __name__=='__main__':check_recipient_exclusion()
