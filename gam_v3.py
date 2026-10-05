"""Penalized additive logistic models with exact numeric values and smoothness.

All axes and labels are fitted within the training fold. Unseen numeric values
interpolate neighboring training coefficients. Test labels are never available.
"""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments_v3'
SMOOTH = ['Annual_Income_USD', 'Daily_Commute_km']


class ExactGAM:
    def __init__(self, ridge=1., smooth=10., commute_smooth=10., gate=False, baseline=False):
        self.ridge = ridge
        self.smooth = smooth
        self.commute_smooth = commute_smooth
        self.gate = gate
        self.baseline = baseline

    def prepare(self, x):
        x = x.copy()
        if self.gate:
            cols = ['Environmental_Concern_Level', 'Subsidy_Available', 'Range_Anxiety_Level']
            x['gate'] = x[cols[0]].astype(str)+'|'+x[cols[1]].astype(str)+'|'+x[cols[2]].astype(str)
        return x

    def matrix(self, x):
        x = self.prepare(x)
        rows = [np.arange(len(x))]; cols = [np.zeros(len(x), dtype=int)]; vals = [np.ones(len(x))]
        for col, axis, start in self.axes:
            v = x[col].to_numpy()
            if col in SMOOTH:
                hi = np.clip(np.searchsorted(axis, v), 0, len(axis)-1)
                lo = np.maximum(hi-1, 0)
                frac = np.divide(v-axis[lo], axis[hi]-axis[lo], out=np.ones(len(v)), where=axis[hi]!=axis[lo])
                frac = np.clip(frac, 0, 1)
                rows.extend([np.arange(len(x)), np.arange(len(x))])
                cols.extend([start+lo, start+hi]); vals.extend([1-frac, frac])
            else:
                codes = pd.Categorical(v, categories=axis).codes.astype(np.int32)
                ok = codes >= 0
                rows.append(np.flatnonzero(ok)); cols.append(start+codes[ok]); vals.append(np.ones(ok.sum()))
        m = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(x), self.n_lookup))
        m.eliminate_zeros()
        if self.baseline:
            m = sparse.hstack([m, self.spline.transform(x[SMOOTH])], format='csr')
        return m

    def fit(self, x, y):
        prepared = self.prepare(x)
        self.axes = []; n = 1
        for col in prepared:
            axis = np.sort(prepared[col].unique())
            self.axes.append((col, axis, n)); n += len(axis)
        self.n_lookup = n
        if self.baseline:
            self.spline = SplineTransformer(n_knots=32, degree=3, knots='quantile', extrapolation='linear', sparse_output=True)
            self.spline.fit(x[SMOOTH])
            n += self.spline.n_features_out_
        self.n_features = n
        X = self.matrix(x)
        ridge = np.full(n, .01); ridge[0] = 0
        dr = []; dc = []; dv = []; row = 0
        for col, axis, start in self.axes:
            if col in SMOOTH:
                ridge[start:start+len(axis)] = self.ridge
                strength = self.smooth if col == SMOOTH[0] else self.commute_smooth
                gap = np.diff(axis).astype(float)
                w = np.sqrt(strength * np.clip(np.median(gap)/gap, .05, 20.))
                for j, weight in enumerate(w):
                    dr.extend([row, row]); dc.extend([start+j, start+j+1]); dv.extend([-weight, weight]); row += 1
            elif col == 'gate':
                ridge[start:start+len(axis)] = 10.
        D = sparse.csr_matrix((dv, (dr, dc)), shape=(row, n))
        P = sparse.diags(ridge, format='csr') + D.T @ D
        diag = np.asarray(X.power(2).sum(axis=0)).ravel()*.1 + P.diagonal()
        scale = 1/np.sqrt(np.maximum(diag, 1e-6))
        initial = np.zeros(n); initial[0] = logit(np.mean(y))/scale[0]
        y = np.asarray(y, dtype=float)

        def loss_gradient(v):
            coef = v*scale
            eta = X @ coef
            pc = P @ coef
            loss = (np.logaddexp(0, eta).sum()-y@eta + .5*coef@pc) / len(y)
            grad = (X.T @ (expit(eta)-y) + pc)*scale / len(y)
            return float(loss), grad

        with threadpool_limits(limits=4):
            result = minimize(loss_gradient, initial, jac=True, method='L-BFGS-B',
                              options=dict(maxiter=700, ftol=1e-13, gtol=2e-9, maxcor=20))
        self.coef_ = result.x*scale
        self.optimization = dict(success=bool(result.success), iterations=int(result.nit),
                                 message=str(result.message), objective=float(result.fun),
                                 max_gradient=float(np.max(np.abs(result.jac))))
        return self

    def predict(self, x):
        return expit(self.matrix(x) @ self.coef_)


CASES = {
    'exact_r1': dict(ridge=1., smooth=0., commute_smooth=0.),
    'smooth_r1_s1': dict(ridge=1., smooth=1.),
    'smooth_r1_s10': dict(ridge=1., smooth=10.),
    'smooth_r1_s100': dict(ridge=1., smooth=100.),
    'smooth_r01_s10': dict(ridge=.1, smooth=10.),
    'smooth_r01_s100': dict(ridge=.1, smooth=100.),
    'smooth_r10_s10': dict(ridge=10., smooth=10.),
    'smooth_r01_s10_gate': dict(ridge=.1, smooth=10., gate=True),
    'smooth_r001_s1': dict(ridge=.01, smooth=1.),
    'smooth_r001_s10': dict(ridge=.01, smooth=10.),
    'spline_r01_s1': dict(ridge=.1, smooth=1., baseline=True),
    'spline_r03_s1': dict(ridge=.3, smooth=1., baseline=True),
    'spline_r1_s10': dict(ridge=1., smooth=10., baseline=True),
    'spline_r3_s10': dict(ridge=3., smooth=10., baseline=True),
    'spline_r1_s0': dict(ridge=1., smooth=0., baseline=True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=list(CASES))
    parser.add_argument('--folds', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT/'data/train.csv'); test = pd.read_csv(ROOT/'data/test.csv')
    base = pd.read_csv(ROOT/'experiments_v2/oof_predictions.csv')
    y = train.Will_Buy_EV.map({'No':0, 'Yes':1}).to_numpy()
    assert train.id.equals(base.id) and np.array_equal(y, base.target)
    x = train.drop(columns=['id', 'Will_Buy_EV']); xt = test.drop(columns='id')
    for fold in args.folds:
        tr = np.flatnonzero(base.fold != fold); va = np.flatnonzero(base.fold == fold)
        for name in args.models:
            dest = OUT/f'gam_{name}_fold{fold}'
            if dest.with_suffix('.npz').exists():
                print(f'Skipping saved {dest.name}', flush=True); continue
            start = time.time(); print(f'Start {dest.name}', flush=True)
            model = ExactGAM(**CASES[name]).fit(x.iloc[tr], y[tr])
            vp = model.predict(x.iloc[va]); tp = model.predict(xt)
            assert np.isfinite(vp).all() and np.isfinite(tp).all()
            result = dict(model='gam_'+name, fold=fold, params=CASES[name],
                          auc=float(roc_auc_score(y[va], vp)), logloss=float(log_loss(y[va], vp)),
                          blend_v2_auc=float(roc_auc_score(y[va], expit((logit(vp)+logit(base.equal_logit_blend.iloc[va]))/2))),
                          optimization=model.optimization, seconds=time.time()-start)
            joblib.dump(vars(model), dest.with_suffix('.joblib'))
            np.savez_compressed(dest.with_suffix('.npz'), indices=va, validation=vp, test=tp)
            dest.with_suffix('.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
