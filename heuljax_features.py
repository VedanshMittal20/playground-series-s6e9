# Adapted from Paul Bryan Elefante (heuljax), KPS6E09 XGB Sample.
# Source: https://www.kaggle.com/code/heuljax/kps6e09-xgb-sample
# Downloaded 2026-09-28. Only feature definitions and data setup are retained.
# Changes: local paths, three saved baseline folds, seed 2026. Training wrapper
# is reference_v3.py; public prediction files are not used.
from __future__ import annotations

import gc
import os
import json
import time
import math
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
import numba
from numba import njit, prange
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import xgboost as xgb

IS_KAGGLE = False
SMOKE_TEST = os.getenv('XGB_SAMPLE_SMOKE', '0') == '1'
FORCE_CPU = os.getenv('XGB_SAMPLE_FORCE_CPU', '0') == '1'
ALLOW_SMOKE_NUMERIC_PARQUET = os.getenv('XGB_SAMPLE_NUMERIC_PARQUET', '0') == '1'
USE_ORIGINAL = True

DATA_DIR = (Path('/kaggle/input/competitions/playground-series-s6e9')
            if IS_KAGGLE else Path(__file__).resolve().parent / 'data')
ORIG_DIR = (Path('/kaggle/input/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety')
            if IS_KAGGLE else Path(__file__).resolve().parent / 'original')
OOF_DIR = Path(__file__).resolve().parent / 'experiments_v3' / 'reference_oof'
TEST_PRED_DIR = OOF_DIR / 'test'
SUBMISSION_PATH = OOF_DIR / 'submission.csv'
EXP_NAME = 'XGB_SAMPLE'
MODEL_NAME = EXP_NAME
if SMOKE_TEST:
    OOF_DIR /= '_smoke'
    TEST_PRED_DIR /= '_smoke'
    SUBMISSION_PATH = Path('_smoke') / 'submission.csv'

TARGET, ID_COL = 'Will_Buy_EV', 'id'
FLOAT_COLS = ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level']
INT_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
            'Charging_Stations_Near_Work']
CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']
RAW_COLS = FLOAT_COLS + INT_COLS + CAT_COLS
NUM_COLS = FLOAT_COLS + INT_COLS
CATEGORY_VOCAB = {
    'Gender': ['Male', 'Female', 'Other'],
    'City_Type': ['Suburban', 'Rural', 'Urban'],
    'Current_Car_Type': ['Sedan', 'SUV', 'Hatchback', 'Truck'],
    'Home_Charging_Possible': ['Yes', 'No'],
    'Subsidy_Available': ['No', 'Yes'],
    'Range_Anxiety_Level': ['Low', 'Medium', 'High'],
}
N_FOLDS, RANDOM_STATE, INNER_FOLDS = 3, 2026, 5
N_THREADS = max(1, min(int(os.getenv('XGB_SAMPLE_THREADS', '8')), os.cpu_count() or 1))
numba.set_num_threads(min(N_THREADS, numba.config.NUMBA_NUM_THREADS))
np.random.seed(RANDOM_STATE)
SMOKE_TRAIN_ROWS, SMOKE_TEST_ROWS = 2400, 256
N_ESTIMATORS = 32 if SMOKE_TEST else 6000
EARLY_STOPPING_ROUNDS = 12 if SMOKE_TEST else 750
GROUP_ESTIMATORS = 16 if SMOKE_TEST else 600
GAM_ITERS = 3 if SMOKE_TEST else 12
GAM_STEP = 0.5
CHUNK_ROWS = 32768
VERBOSE_EVERY = 0 if SMOKE_TEST else 100
USE_RATE_MONOTONICITY = True
INCOME_ALPHAS, COMMUTE_ALPHAS = (5., 20., 50.), (20., 100.)
INCOME_WIDTHS, COMMUTE_WIDTHS = (1, 2, 5, 10, 25, 50, 100, 250), (1, 2, 5, 10, 25, 50)
HIER_WIDTHS, HIER_ALPHAS = (5, 25, 100), (2., 5., 10., 20., 50.)
DETAIL_PAIRS = ((1, 5), (5, 25), (25, 100), (100, 250))
LATENT_ALPHAS = (2., 5., 10., 20., 50., 100.)
MSTE_ALPHAS = (1., 2., 5., 10., 50., 200.)
MIX_WIDTHS = (16, 64, 256)
SHIFT_GRID = np.linspace(-16., 16., 129)
OTHER_COLS = ['Age', 'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
              'Charging_Stations_Near_Work', 'Gender', 'City_Type',
              'Current_Car_Type', 'Home_Charging_Possible']

REQUIRED_DEPENDENCIES = (
    'numpy>=1.26', 'pandas>=2.0', 'scipy>=1.11', 'scikit-learn>=1.3',
    'numba>=0.59', 'xgboost>=2.0', 'threadpoolctl>=3.1', 'pyarrow or fastparquet',
)
PARQUET_ENGINE = next((name for name in ('pyarrow', 'fastparquet')
                       if importlib.util.find_spec(name) is not None), None)
if PARQUET_ENGINE is None and not (SMOKE_TEST and ALLOW_SMOKE_NUMERIC_PARQUET):
    raise ImportError('Install pyarrow or fastparquet before running this notebook.')
if PARQUET_ENGINE is None:
    if importlib.util.find_spec('thrift') is None:
        raise ImportError('The numeric smoke-test writer requires thrift.')
    print('SMOKE TEST | numeric Parquet writer')

def detect_device():
    if FORCE_CPU:
        print('DEVICE=cpu | FORCE_CPU=True')
        return 'cpu'
    probe_x = np.random.default_rng(RANDOM_STATE).normal(size=(128, 4)).astype(np.float32)
    probe_y = (probe_x[:, 0] > 0).astype(np.float32)
    try:
        with warnings.catch_warnings(record=True):
            probe = xgb.train(
                dict(objective='binary:logistic', device='cuda', tree_method='hist',
                     max_depth=1, nthread=N_THREADS),
                xgb.DMatrix(probe_x, label=probe_y, nthread=N_THREADS), 1)
        actual = json.loads(probe.save_config())['learner']['generic_param']['device']
        del probe
        if str(actual).startswith('cuda'):
            return actual
    except Exception as exc:
        print(f'CUDA unavailable: {type(exc).__name__}')
    print('DEVICE=cpu | CUDA unavailable')
    return 'cpu'

DEVICE = detect_device()
XGB_PARAMS = dict(objective='binary:logistic', eval_metric='auc', base_score=0.5,
                  tree_method='hist', device=DEVICE, learning_rate=0.015,
                  max_depth=5, max_bin=256, colsample_bynode=0.8,
                  colsample_bytree=1.0, subsample=1.0, min_child_weight=1.0,
                  reg_alpha=10.0, reg_lambda=15.0, nthread=N_THREADS)
GROUP_PARAMS = dict(objective='reg:logistic', tree_method='hist', device='cpu',
                    learning_rate=0.03, max_depth=3, min_child_weight=20.0,
                    subsample=0.8, colsample_bynode=0.8, reg_alpha=2.0,
                    reg_lambda=100.0, n_jobs=N_THREADS, n_estimators=GROUP_ESTIMATORS)
for directory in (OOF_DIR, TEST_PRED_DIR, SUBMISSION_PATH.parent):
    directory.mkdir(parents=True, exist_ok=True)
print(f"{EXP_NAME} | {'SMOKE' if SMOKE_TEST else 'CV'} | {DEVICE} | folds={N_FOLDS}")


SOURCE_COLS = ['ORIG_INC_Y', 'ORIG_INC_COUNT_LOG', 'ORIG_INC_SEEN', 'ORIG_INC_LOCAL_Y']
BASE_FOLD_COLS = (
    [f'TE_INC_A{a:g}' for a in INCOME_ALPHAS]
    + [f'TE_CMT_A{a:g}' for a in COMMUTE_ALPHAS]
    + ['SUPPORT_INC_LOG', 'SUPPORT_CMT_LOG']
    + [f'NBR_INC_W{w}' for w in INCOME_WIDTHS]
    + [f'NBR_DETAIL_W{a}_{b}' for a, b in DETAIL_PAIRS]
    + [f'HIER_INC_W{w}_A{a:g}' for w in HIER_WIDTHS for a in HIER_ALPHAS]
    + [f'INC_SURPRISE_W{w}' for w in HIER_WIDTHS]
    + [f'NBR_CMT_W{w}' for w in COMMUTE_WIDTHS]
    + ['DCMT_LOCAL_W5', 'P_DCMT_LOCAL_W5'])
GXP_COLS = ['GXP_D3_R100'] + [f'GXP_D3_R100_A{a:g}' for a in (2., 5., 10., 20., 50.)]
LATENT_NAMES = ['CONCERN', 'SUBSIDY', 'ANXIETY_PENALTY', 'BUY_SCORE']
LAT_COLS = [f'LAT_INC_{n}_A{a:g}' for n in LATENT_NAMES for a in LATENT_ALPHAS]
LAT_COLS += ['LAT_INC_CONCERN_DEV', 'LAT_INC_SUBSIDY_DEV', 'LAT_INC_ANXIETY_DEV', 'LAT_INC_SCORE_DEV']
DIGIT_COLS = [f'DIG_INC_10E{k}' for k in range(6)] + [f'DIG_CMT10_10E{k}' for k in range(4)]
WORRY_COLS = ['WTE05_A5', 'WHIER05_W25_A20']
UNCERTAINTY_COLS = ['GXP_POST_SD_A20', 'GXP_POST_LO_A20', 'GXP_POST_HI_A20']
MIX_COLS = [f'MIX_W{w}_{s}' for w in MIX_WIDTHS
            for s in ('PRIOR', 'POST20', 'POST100', 'ROW_PROB', 'INFO', 'GAIN')]
CHANNELS = ['GXP_D3_R100', 'GXP_D3_R100_A20', 'GXP_D3_R100_A50']
CHANNELS += [f'MIX_W{w}_POST100' for w in MIX_WIDTHS]
COMPOSITION_COLS = [f'CSHIFT_{c}' for c in CHANNELS] + [f'CCORR_{c}' for c in CHANNELS]
COORD_COLS = ['GAM_GATE_COORD', 'GAM_INCOME_COORD', 'GAM_COMMUTE_COORD', 'GAM_OTHER_COORD']
MSTE_KEYS = ['INC10', 'INC100', 'INC1K', 'CMTINT']
MSTE_COLS = [c for key in MSTE_KEYS
             for c in ([f'MSTE_{key}_A{a:g}' for a in MSTE_ALPHAS] + [f'MSTE_{key}_LOGN'])]
FEATURE_COLS = (RAW_COLS + SOURCE_COLS + BASE_FOLD_COLS + GXP_COLS + LAT_COLS + DIGIT_COLS
                + WORRY_COLS + UNCERTAINTY_COLS + MIX_COLS + COMPOSITION_COLS + COORD_COLS + MSTE_COLS)
assert len(FEATURE_COLS) == len(set(FEATURE_COLS)) == 173
assert not {TARGET, ID_COL, 'Buyer_ID'}.intersection(FEATURE_COLS)
COL = {name: i for i, name in enumerate(FEATURE_COLS)}
FEATURE_TYPES = ['c' if c in CAT_COLS + DIGIT_COLS else 'q' for c in FEATURE_COLS]
RATE_COLS = [c for c in FEATURE_COLS if
            c in ('ORIG_INC_Y', 'ORIG_INC_LOCAL_Y', 'WTE05_A5', 'WHIER05_W25_A20',
                  'GXP_POST_LO_A20', 'GXP_POST_HI_A20')
            or c.startswith(('TE_INC_', 'TE_CMT_', 'NBR_INC_', 'NBR_CMT_', 'HIER_INC_', 'GXP_D3_R100_A'))
            or (c.startswith('MIX_') and any(s in c for s in ('_PRIOR', '_POST')))
            or (c.startswith('MSTE_') and '_A' in c)]
if USE_RATE_MONOTONICITY:
    XGB_PARAMS['monotone_constraints'] = '(' + ','.join('1' if c in RATE_COLS else '0' for c in FEATURE_COLS) + ')'


def box_sum(values, width):
    values = np.asarray(values, dtype=np.float64)
    prefix = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)], axis=0)
    center = np.arange(len(values))
    return prefix[np.minimum(center + width + 1, len(values))] - prefix[np.maximum(center - width, 0)]

GAUSSIAN_BACKENDS = {16: 'direct', 64: 'fft', 256: 'fft'}

def gaussian_smooth(values, width, backend=None):
    backend = GAUSSIAN_BACKENDS[int(width)] if backend is None else backend
    if backend == 'direct':
        return gaussian_filter1d(values, width, axis=0, mode='constant')
    radius = int(4.0 * width + 0.5)
    positions = np.arange(-radius, radius+1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / width)**2)
    kernel /= kernel.sum()
    return fftconvolve(values, kernel[:, None], mode='same', axes=0)

@njit(cache=False)
def gam_kernel(codes, sizes, ridges, widths, targets, iterations, step):
    n, nt = codes.shape
    prior = min(max(np.mean(targets), 1e-6), 1.0-1e-6)
    intercept = math.log(prior/(1.0-prior))
    eta = np.full(n, intercept, dtype=np.float64)
    tables = np.zeros((nt, np.max(sizes)), dtype=np.float64)
    for epoch in range(iterations):
        for term in range(nt):
            length = sizes[term]
            grad = np.zeros(length, dtype=np.float64)
            hess = np.zeros(length, dtype=np.float64)
            for i in range(n):
                e = max(min(eta[i], 700.0), -700.0)
                p = 1.0 / (1.0 + math.exp(-e))
                code = codes[i, term]
                grad[code] += targets[i] - p
                hess[code] += p * (1.0-p)
            width = widths[term]
            if width:
                pg = np.zeros(length+1, dtype=np.float64)
                ph = np.zeros(length+1, dtype=np.float64)
                for j in range(length):
                    pg[j+1] = pg[j] + grad[j]
                    ph[j+1] = ph[j] + hess[j]
                for j in range(length):
                    lo, hi = max(0, j-width), min(length, j+width+1)
                    grad[j] = pg[hi] - pg[lo]
                    hess[j] = ph[hi] - ph[lo]
            delta = step * grad / (hess + ridges[term])
            for j in range(length):
                tables[term, j] += delta[j]
            for i in range(n):
                eta[i] += delta[codes[i, term]]
    return intercept, tables, eta

@njit(cache=False, parallel=True)
def mixture_kernel(counts, q0, q1, mu):
    ng, nk = counts.shape
    pi = np.empty(ng, dtype=np.float64)
    information = np.empty(ng, dtype=np.float64)
    gain = np.empty(ng, dtype=np.float64)
    for i in prange(ng):
        p = mu[i]
        for iteration in range(24):
            g = 10.0 * (mu[i]/p - (1.0-mu[i])/(1.0-p))
            h = 10.0 * (mu[i]/(p*p) + (1.0-mu[i])/((1.0-p)*(1.0-p)))
            for j in range(nk):
                d = q1[i, j] - q0[i, j]
                m = max(q0[i, j] + p*d, 1e-300)
                r = d/m
                g += counts[i, j] * r
                h += counts[i, j] * r*r
            step = max(-0.15, min(0.15, g/max(h, 1e-8)))
            p = max(1e-4, min(1.0-1e-4, p+step))
        pi[i] = p
        ii, gg, total = 0.0, 0.0, 0.0
        for j in range(nk):
            d = q1[i, j] - q0[i, j]
            m = max(q0[i, j] + p*d, 1e-300)
            old = max(q0[i, j] + mu[i]*d, 1e-300)
            ii += counts[i, j]*(d/m)**2
            gg += counts[i, j]*math.log(m/old)
            total += counts[i, j]
        information[i] = math.log1p(ii)
        gain[i] = gg/max(total, 1.0)
    return pi, information, gain

@njit(cache=False, parallel=True)
def composition_curve_kernel(nuisance_sorted, boundaries, grid_start, grid_step, n_grid):
    ng = len(boundaries)-1
    curve = np.zeros((ng, n_grid), dtype=np.float64)
    ratio = math.exp(-grid_step)
    for group in prange(ng):
        lo, hi = boundaries[group], boundaries[group+1]
        for i in range(lo, hi):
            v = math.exp(min(700.0, max(-700.0, -nuisance_sorted[i]-grid_start)))
            for k in range(n_grid):
                curve[group, k] += 1.0/(1.0+v)
                v *= ratio
        for k in range(n_grid):
            curve[group, k] /= max(hi-lo, 1)
    return curve

@njit(cache=False, parallel=True)
def invert_curve_kernel(curves, indices, probabilities, grid):
    n, nc = probabilities.shape
    out = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        idx = indices[i]
        for j in range(nc):
            p = probabilities[i, j]
            if p <= curves[idx, 0]:
                out[i, j] = grid[0]
            elif p >= curves[idx, -1]:
                out[i, j] = grid[-1]
            else:
                lo, hi = 0, len(grid)-1
                while hi-lo > 1:
                    mid = (lo+hi)//2
                    if curves[idx, mid] < p:
                        lo = mid
                    else:
                        hi = mid
                denom = curves[idx, hi] - curves[idx, lo]
                frac = (p-curves[idx, lo])/denom if denom > 1e-14 else 0.0
                out[i, j] = grid[lo] + frac*(grid[hi]-grid[lo])
    return out

class ValueAxis:
    def __init__(self, values):
        self.values, codes = np.unique(np.asarray(values), return_inverse=True)
        self.codes = codes.astype(np.int32)
        self.size = len(self.values)
        if self.size == 0:
            raise ValueError('Empty donor value axis')

    def locate(self, query):
        query = np.asarray(query)
        pos = np.searchsorted(self.values, query)
        hi = np.minimum(pos, self.size-1)
        lo = np.maximum(pos-1, 0)
        exact = (pos < self.size) & (self.values[hi] == query)
        nearest = np.where(np.abs(query-self.values[lo]) <= np.abs(query-self.values[hi]), lo, hi)
        return np.where(exact, hi, nearest).astype(np.int32), exact

class TargetTable:
    def __init__(self, values, targets):
        self.axis = ValueAxis(values)
        self.count = np.bincount(self.axis.codes, minlength=self.axis.size).astype(np.float64)
        self.total = np.bincount(self.axis.codes, weights=targets, minlength=self.axis.size)
        self.prior = float(np.mean(targets))
        self.local_cache = {}

    def lookup(self, query):
        idx, exact = self.axis.locate(query)
        return idx, exact, self.count[idx]*exact, self.total[idx]*exact

    def local(self, width, alpha=20.0):
        key = (width, float(alpha))
        if key not in self.local_cache:
            count = np.maximum(box_sum(self.count, width)-self.count, 0.0)
            total = box_sum(self.total, width)-self.total
            self.local_cache[key] = (total+alpha*self.prior)/(count+alpha)
        return self.local_cache[key]

    def rate(self, query, alpha):
        _, _, count, total = self.lookup(query)
        return (total+alpha*self.prior)/(count+alpha)


train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
original = (pd.read_csv(ORIG_DIR / 'EV_Adoption_and_Range_Anxiety_Dataset.csv')
            if USE_ORIGINAL else None)

for frame, required in ((train, [ID_COL, TARGET] + RAW_COLS), (test, [ID_COL] + RAW_COLS)):
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Missing columns: {set(required) - set(frame.columns)}')
    if not frame[ID_COL].is_unique or frame[ID_COL].isna().any():
        raise ValueError('IDs must be unique and nonmissing')
if TARGET in test.columns:
    raise ValueError('Test data must not contain target labels')
if not set(train[TARGET].unique()).issubset({'No', 'Yes'}):
    raise ValueError('Target must be No/Yes without missing values')
if len(np.intersect1d(train[ID_COL].to_numpy(), test[ID_COL].to_numpy())):
    raise ValueError('Train and test IDs overlap')
if not len(test):
    raise ValueError('Test data is empty')

def feature_hashes(frame):
    normalized = frame[RAW_COLS].copy()
    for c in NUM_COLS:
        normalized[c] = pd.to_numeric(normalized[c], errors='coerce').astype(np.float64)
    for c in CAT_COLS:
        normalized[c] = normalized[c].astype('string').fillna('__MISSING__')
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy(np.uint64)

if USE_ORIGINAL:
    if not set(RAW_COLS + [TARGET]).issubset(original.columns):
        raise ValueError('Original dataset schema mismatch')
    original_hash = feature_hashes(original)
    train_hash = feature_hashes(train)
    test_hash = feature_hashes(test)
    keep = (~np.isin(original_hash, train_hash)
            & ~np.isin(original_hash, test_hash)
            & original[TARGET].isin(['No', 'Yes']).to_numpy())
    original = original.loc[keep].reset_index(drop=True)
    del original_hash, train_hash, test_hash, keep
    if not len(original):
        raise ValueError('No eligible original rows remain')

train_row_index = np.arange(len(train), dtype=np.int64)
if SMOKE_TEST:
    if len(train) > SMOKE_TRAIN_ROWS:
        chosen, _ = train_test_split(train_row_index, train_size=SMOKE_TRAIN_ROWS,
                                     stratify=train[TARGET], random_state=RANDOM_STATE)
        chosen.sort()
        train_row_index = chosen
        train = train.iloc[chosen].reset_index(drop=True)
    test = test.iloc[:SMOKE_TEST_ROWS].copy().reset_index(drop=True)
TRAIN_IDS, TEST_IDS = train[ID_COL].to_numpy(copy=True), test[ID_COL].to_numpy(copy=True)
y = train[TARGET].eq('Yes').to_numpy(np.int8)
if np.bincount(y, minlength=2).min() < N_FOLDS:
    raise ValueError('Insufficient observations per class for ten stratified folds')

def pack_raw(frame):
    out = {}
    for name in NUM_COLS:
        out[name] = pd.to_numeric(frame[name], errors='coerce').to_numpy(np.float64)
    for name in CAT_COLS:
        categories = CATEGORY_VOCAB[name]
        codes = pd.Categorical(frame[name].astype('string'), categories=categories).codes.astype(np.int16)
        out[name] = np.where(codes < 0, len(categories), codes).astype(np.int16)
    return out

def take_raw(raw, rows):
    return {k: np.ascontiguousarray(v[rows]) for k, v in raw.items()}

def fit_medians(raw):
    medians = {c: float(np.nanmedian(np.where(np.isfinite(raw[c]), raw[c], np.nan))) for c in NUM_COLS}
    if not all(np.isfinite(v) for v in medians.values()):
        raise ValueError('Entirely missing numeric donor column')
    return medians

def prepare_raw(raw, medians):
    result = {c: np.where(np.isfinite(raw[c]), raw[c], medians[c]).astype(np.float64, copy=False)
              for c in NUM_COLS}
    result.update({c: raw[c] for c in CAT_COLS})
    return result

def latent_values(raw):
    income = raw['Annual_Income_USD']
    concern = raw['Environmental_Concern_Level']
    subsidy = (raw['Subsidy_Available'] == CATEGORY_VOCAB['Subsidy_Available'].index('Yes')).astype(np.float64)
    anxiety = raw['Range_Anxiety_Level']
    penalty = np.where(anxiety == 1, 1.0, np.where(anxiety == 2, 3.0, 0.0))
    score = 1.2*income/100000.0 + 0.6*concern + 2.0*subsidy - penalty
    return np.column_stack([concern, subsidy, penalty, score])

def gate_codes(raw):
    concern = np.rint(raw['Environmental_Concern_Level']).astype(np.int32)-1
    anxiety = raw['Range_Anxiety_Level'].astype(np.int32)
    subsidy = (raw['Subsidy_Available'] == 1).astype(np.int32)
    valid = (concern >= 0) & (concern < 5) & (anxiety < 3) & (raw['Subsidy_Available'] < 2)
    return np.where(valid, concern*6 + subsidy*3 + anxiety, 30).astype(np.int32)

def worry_key(raw):
    home = (raw['Home_Charging_Possible'] == 0).astype(np.float64)
    value = (raw['Daily_Commute_km'] - 5.0*raw['Charging_Stations_Near_Home']
             - 5.0*raw['Charging_Stations_Near_Work'] - 150.0*home)
    return np.rint(value/0.5).astype(np.int64)

def mste_keys(raw):
    income, commute = raw['Annual_Income_USD'], raw['Daily_Commute_km']
    return dict(INC10=np.floor(income/10), INC100=np.floor(income/100),
                INC1K=np.floor(income/1000), CMTINT=np.floor(commute))

class SourceIncome:
    def __init__(self, source):
        if source is None:
            self.enabled = False
            self.prior = 0.5
            return
        self.enabled = True
        yy = source[TARGET].eq('Yes').to_numpy(np.float64)
        self.prior = float(yy.mean())
        vals = pd.to_numeric(source['Annual_Income_USD'], errors='coerce').to_numpy(np.float64)
        good = np.isfinite(vals)
        if not good.any():
            raise ValueError('Original income is entirely missing')
        self.table = TargetTable(vals[good], yy[good])
        self.mean = self.table.total/self.table.count
        self.local = ((box_sum(self.table.total, 25)+20.0*self.prior)
                      /(box_sum(self.table.count, 25)+20.0))

    def transform(self, income):
        if not self.enabled:
            return np.column_stack([np.full(len(income), 0.5), np.zeros(len(income)),
                                    np.zeros(len(income)), np.full(len(income), 0.5)])
        idx, exact, count, total = self.table.lookup(income)
        mean = np.where(exact, self.mean[idx], self.prior)
        return np.column_stack([mean, np.log1p(count), exact.astype(np.float64), self.local[idx]])

RAW_TRAIN, RAW_TEST = pack_raw(train), pack_raw(test)
SOURCE = SourceIncome(original)
del train, test, original
print(f'TRAIN={len(y):,} TEST={len(TEST_IDS):,} FEATURES={len(FEATURE_COLS)}')


class AdditiveState:
    def __init__(self, raw, targets, axes, main_only=False):
        self.axes = axes
        self.terms = [('gate', 0, 5.0, 0)]
        if not main_only:
            self.terms += [('income', w, 25.0, 1) for w in (0, 2, 8, 32, 128)]
            self.terms += [('commute', w, 25.0, 2) for w in (0, 5)]
        self.terms += [(c, 0, 5.0, 3) for c in OTHER_COLS]
        arrays, sizes = [], []
        for key, width, ridge, component in self.terms:
            if key == 'gate':
                arrays.append(gate_codes(raw))
                sizes.append(31)
            else:
                arrays.append(axes[key].codes)
                sizes.append(axes[key].size)
        codes = np.ascontiguousarray(np.column_stack(arrays), dtype=np.int32)
        self.intercept, self.tables, self.fitted_eta = gam_kernel(
            codes, np.asarray(sizes, np.int32),
            np.asarray([t[2] for t in self.terms], np.float64),
            np.asarray([t[1] for t in self.terms], np.int32),
            np.asarray(targets, np.float64), GAM_ITERS, GAM_STEP)

    def components(self, raw):
        n = len(raw['Age'])
        result = np.zeros((n, 4), dtype=np.float64)
        result[:, 3] = self.intercept
        located = {}
        for t, (key, width, ridge, component) in enumerate(self.terms):
            if key not in located:
                if key == 'gate':
                    located[key] = (gate_codes(raw), np.ones(n, dtype=bool))
                else:
                    column = {'income': 'Annual_Income_USD', 'commute': 'Daily_Commute_km'}.get(key, key)
                    located[key] = self.axes[key].locate(raw[column])
            idx, exact = located[key]
            effect = self.tables[t, idx]
            result[:, component] += effect if width else effect*exact
        return result

class IncomeGroupPrior:
    def __init__(self, raw, targets, income_table, source, seed):
        axis, counts, total = income_table.axis, income_table.count, income_table.total
        latent = latent_values(raw)
        score = latent[:, 3]
        commute = raw['Daily_Commute_km']
        values = [latent[:, 0], latent[:, 1], latent[:, 2], score, score**2,
                  expit(score-5.5), expit(2.0*(score-5.5)), expit(3.0*(score-5.5)),
                  (raw['Home_Charging_Possible'] == 0).astype(np.float64), commute, commute**2,
                  raw['Charging_Stations_Near_Home'], raw['Charging_Stations_Near_Work'],
                  raw['Age'], raw['Number_of_Cars_Owned']]
        values += [(score >= threshold).astype(np.float64) for threshold in (4.5, 5.0, 5.5, 6.0, 6.5)]
        for column in ('City_Type', 'Current_Car_Type', 'Gender'):
            values += [(raw[column] == CATEGORY_VOCAB[column].index(category)).astype(np.float64)
                       for category in sorted(CATEGORY_VOCAB[column])]
        base = [axis.values/100000.0, np.log1p(counts)]
        base += [np.bincount(axis.codes, weights=v, minlength=axis.size)/counts for v in values]
        base += list(source.transform(axis.values).T)
        matrix = np.column_stack(base).astype(np.float64)
        self.template = matrix.mean(axis=0)
        self.mean = self.template.copy()
        self.scale = matrix.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        matrix = np.ascontiguousarray((matrix-self.mean)/self.scale)
        parameters = dict(GROUP_PARAMS, random_state=seed)
        self.model = xgb.XGBRegressor(**parameters)
        self.model.fit(matrix, total/counts, sample_weight=np.sqrt(counts), verbose=False)
        self.prediction = np.clip(self.model.predict(matrix), 1e-5, 1.0-1e-5).astype(np.float64)
        self.source = source

    def transform(self, income, idx, exact):
        prior = self.prediction[idx].copy()
        if not exact.all():
            unseen, reverse = np.unique(income[~exact], return_inverse=True)
            matrix = np.tile(self.template, (len(unseen), 1))
            matrix[:, 0] = unseen/100000.0
            matrix[:, 1] = 0.0
            matrix[:, -4:] = self.source.transform(unseen)
            scaled = np.ascontiguousarray((matrix-self.mean)/self.scale)
            prior[~exact] = np.clip(self.model.predict(scaled), 1e-5, 1.0-1e-5)[reverse]
        return prior

class DonorState:
    def __init__(self, raw, targets, source, seed):
        targets = np.asarray(targets, dtype=np.float64)
        if len(raw['Age']) != len(targets) or len(targets) == 0 or len(np.unique(targets)) != 2:
            raise ValueError('Donor must contain aligned data and both target classes')
        self.medians = fit_medians(raw)
        raw = prepare_raw(raw, self.medians)
        self.source = source
        self.prior = float(targets.mean())
        self.income = TargetTable(raw['Annual_Income_USD'], targets)
        self.commute = TargetTable(raw['Daily_Commute_km'], targets)
        self.worry = TargetTable(worry_key(raw), targets)
        self.mste = {key: TargetTable(value, targets) for key, value in mste_keys(raw).items()}
        axes = {'income': self.income.axis, 'commute': self.commute.axis}
        axes.update({c: ValueAxis(raw[c]) for c in OTHER_COLS})
        self.gam = AdditiveState(raw, targets, axes)
        main = AdditiveState(raw, targets, axes, main_only=True)
        main_probability = expit(main.fitted_eta)
        cg = self.commute.axis.codes
        gradient = np.bincount(cg, weights=targets-main_probability, minlength=self.commute.axis.size)
        hessian = np.bincount(cg, weights=main_probability*(1.0-main_probability), minlength=self.commute.axis.size)
        self.commute_effect = box_sum(gradient, 5)/(box_sum(hessian, 5)+20.0)
        self.main = main
        self.gxp = IncomeGroupPrior(raw, targets, self.income, source, seed)
        self.latent_prior = latent_values(raw).mean(axis=0)
        self.latent_total = np.column_stack([
            np.bincount(self.income.axis.codes, weights=v, minlength=self.income.axis.size)
            for v in latent_values(raw).T])
        income_code = self.income.axis.codes
        gate = gate_codes(raw)
        self.n_gate = max(30, int(gate.max())+1)
        size = self.income.axis.size*self.n_gate
        joint_code = income_code.astype(np.int64)*self.n_gate + gate
        c1 = np.bincount(joint_code, weights=targets, minlength=size).reshape(-1, self.n_gate)
        allcounts = np.bincount(joint_code, minlength=size).reshape(-1, self.n_gate).astype(np.float64)
        c0 = allcounts-c1
        marg0 = (c0.sum(axis=0)+0.5)/(c0.sum()+0.5*self.n_gate)
        marg1 = (c1.sum(axis=0)+0.5)/(c1.sum()+0.5*self.n_gate)
        self.marg0, self.marg1 = marg0, marg1
        self.mixtures = {}
        combined = np.column_stack([c0, c1])
        for width in MIX_WIDTHS:
            smooth = gaussian_smooth(combined, width)*np.sqrt(2.0*np.pi)*width
            s0 = np.maximum(smooth[:, :self.n_gate]-c0, 0.0)
            s1 = np.maximum(smooth[:, self.n_gate:]-c1, 0.0)
            q0 = (s0+30.0*marg0)/(s0.sum(axis=1, keepdims=True)+30.0)
            q1 = (s1+30.0*marg1)/(s1.sum(axis=1, keepdims=True)+30.0)
            mu = np.clip((s1.sum(axis=1)+20.0*self.prior)
                         /(s0.sum(axis=1)+s1.sum(axis=1)+20.0), 1e-4, 1.0-1e-4)
            pi, information, gain = mixture_kernel(allcounts, q0, q1, mu)
            denominator = q0 + pi[:, None]*(q1-q0)
            row_probability = pi[:, None]*q1/np.maximum(denominator, 1e-300)
            self.mixtures[width] = dict(pi=pi, row_probability=row_probability,
                                        information=information, gain=gain)
        donor_components = self.gam.components(raw)
        nuisance = donor_components[:, 0] + donor_components[:, 3] + 0.25*donor_components[:, 2]
        order = np.argsort(income_code, kind='stable')
        boundaries = np.r_[0, np.cumsum(self.income.count.astype(np.int64))]
        curve = composition_curve_kernel(nuisance[order], boundaries, SHIFT_GRID[0],
                                         SHIFT_GRID[1]-SHIFT_GRID[0], len(SHIFT_GRID))
        global_curve = np.average(curve, weights=self.income.count, axis=0)
        self.curves = np.vstack([curve, global_curve])
        if not np.isfinite(self.curves).all() or np.any(np.diff(self.curves, axis=1) < -1e-12):
            raise AssertionError('Invalid composition curves')
        self.gam.fitted_eta = None
        self.main.fitted_eta = None
        for axis in list(axes.values()) + [self.worry.axis] + [t.axis for t in self.mste.values()]:
            axis.codes = None
        for width in INCOME_WIDTHS:
            self.income.local(width)
        for width in COMMUTE_WIDTHS:
            self.commute.local(width)
        self.worry.local(25)

    def transform(self, query):
        raw = prepare_raw(query, self.medians)
        n = len(raw['Age'])
        result = np.empty((n, len(FEATURE_COLS)), dtype=np.float32)
        for c in RAW_COLS:
            result[:, COL[c]] = raw[c]
        income, commute = raw['Annual_Income_USD'], raw['Daily_Commute_km']
        result[:, [COL[c] for c in SOURCE_COLS]] = self.source.transform(income)
        ii, ie, count, total = self.income.lookup(income)
        ci, ce, ccount, ctotal = self.commute.lookup(commute)
        for alpha in INCOME_ALPHAS:
            result[:, COL[f'TE_INC_A{alpha:g}']] = (total+alpha*self.prior)/(count+alpha)
        for alpha in COMMUTE_ALPHAS:
            result[:, COL[f'TE_CMT_A{alpha:g}']] = (ctotal+alpha*self.prior)/(ccount+alpha)
        result[:, COL['SUPPORT_INC_LOG']] = np.log1p(count)
        result[:, COL['SUPPORT_CMT_LOG']] = np.log1p(ccount)
        local = {w: self.income.local(w)[ii] for w in INCOME_WIDTHS}
        for width in INCOME_WIDTHS:
            result[:, COL[f'NBR_INC_W{width}']] = local[width]
        for a, b in DETAIL_PAIRS:
            result[:, COL[f'NBR_DETAIL_W{a}_{b}']] = local[a]-local[b]
        for width in HIER_WIDTHS:
            for alpha in HIER_ALPHAS:
                result[:, COL[f'HIER_INC_W{width}_A{alpha:g}']] = (total+alpha*local[width])/(count+alpha)
            result[:, COL[f'INC_SURPRISE_W{width}']] = result[:, COL['TE_INC_A5']]-local[width]
        for width in COMMUTE_WIDTHS:
            result[:, COL[f'NBR_CMT_W{width}']] = self.commute.local(width)[ci]
        main = self.main.components(raw).sum(axis=1)
        effect = self.commute_effect[ci]
        result[:, COL['DCMT_LOCAL_W5']] = effect
        result[:, COL['P_DCMT_LOCAL_W5']] = expit(main+effect)
        prior = self.gxp.transform(income, ii, ie)
        result[:, COL['GXP_D3_R100']] = prior
        for alpha in (2., 5., 10., 20., 50.):
            result[:, COL[f'GXP_D3_R100_A{alpha:g}']] = (total+alpha*prior)/(count+alpha)
        latent = latent_values(raw)
        grouped = self.latent_total[ii]*ie[:, None]
        for j, name in enumerate(LATENT_NAMES):
            for alpha in LATENT_ALPHAS:
                result[:, COL[f'LAT_INC_{name}_A{alpha:g}']] = (grouped[:, j]+alpha*self.latent_prior[j])/(count+alpha)
        for j, name in enumerate(('CONCERN', 'SUBSIDY', 'ANXIETY', 'SCORE')):
            expectation = (grouped[:, j]+20.0*self.latent_prior[j])/(count+20.0)
            result[:, COL[f'LAT_INC_{name}_DEV']] = latent[:, j]-expectation
        integer_income, integer_commute = np.rint(income).astype(np.int64), np.rint(10.0*commute).astype(np.int64)
        for k in range(6):
            result[:, COL[f'DIG_INC_10E{k}']] = (integer_income//10**k) % 10
        for k in range(4):
            result[:, COL[f'DIG_CMT10_10E{k}']] = (integer_commute//10**k) % 10
        wi, we, wc, wt = self.worry.lookup(worry_key(raw))
        result[:, COL['WTE05_A5']] = (wt+5.0*self.prior)/(wc+5.0)
        result[:, COL['WHIER05_W25_A20']] = (wt+20.0*self.worry.local(25)[wi])/(wc+20.0)
        p20 = (total+20.0*prior)/(count+20.0)
        sd = np.sqrt(np.maximum(p20*(1.0-p20)/(count+21.0), 1e-12))
        for c, value in zip(UNCERTAINTY_COLS, (sd, p20-sd, p20+sd)):
            result[:, COL[c]] = value
        gg = gate_codes(raw)
        safe_gate = np.minimum(gg, self.n_gate-1)
        known_gate = gg < self.n_gate
        unknown_row_probability = self.prior*self.marg1[safe_gate]/np.maximum(
            self.marg0[safe_gate]+self.prior*(self.marg1[safe_gate]-self.marg0[safe_gate]), 1e-300)
        for width in MIX_WIDTHS:
            state = self.mixtures[width]
            pi = np.where(ie, state['pi'][ii], self.prior)
            row_p = np.where(ie, state['row_probability'][ii, safe_gate], unknown_row_probability)
            row_p = np.where(known_gate, row_p, pi)
            values = (pi, (total+20.0*pi)/(count+20.0), (total+100.0*pi)/(count+100.0),
                      row_p, np.where(ie, state['information'][ii], 0.0), np.where(ie, state['gain'][ii], 0.0))
            for suffix, value in zip(('PRIOR', 'POST20', 'POST100', 'ROW_PROB', 'INFO', 'GAIN'), values):
                result[:, COL[f'MIX_W{width}_{suffix}']] = value
        components = self.gam.components(raw)
        result[:, [COL[c] for c in COORD_COLS]] = components
        channel_probabilities = np.asarray(result[:, [COL[c] for c in CHANNELS]], dtype=np.float64)
        curve_index = np.where(ie, ii, self.income.axis.size).astype(np.int32)
        shift = invert_curve_kernel(self.curves, curve_index, channel_probabilities, SHIFT_GRID)
        result[:, [COL[c] for c in COMPOSITION_COLS[:6]]] = shift
        result[:, [COL[c] for c in COMPOSITION_COLS[6:]]] = shift-0.25*components[:, 1, None]
        for key, values in mste_keys(raw).items():
            _, _, cn, sy = self.mste[key].lookup(values)
            for alpha in MSTE_ALPHAS:
                result[:, COL[f'MSTE_{key}_A{alpha:g}']] = (sy+alpha*self.prior)/(cn+alpha)
            result[:, COL[f'MSTE_{key}_LOGN']] = np.log1p(cn)
        margin = (components[:, 0]+components[:, 3]+0.25*(components[:, 1]+components[:, 2])).astype(np.float32)
        if not np.isfinite(result).all() or not np.isfinite(margin).all():
            raise AssertionError('Nonfinite features or initial margin')
        return result, margin

    def transform_into(self, raw, rows, feature_buffer, margin_buffer, positions=None):
        rows = np.asarray(rows, dtype=np.int64)
        if positions is None:
            positions = np.arange(len(rows))
        for start in range(0, len(rows), CHUNK_ROWS):
            stop = min(start+CHUNK_ROWS, len(rows))
            values, margins = self.transform(take_raw(raw, rows[start:stop]))
            feature_buffer[positions[start:stop]] = values
            margin_buffer[positions[start:stop]] = margins


def fit_donor_state(donor, receiving, outer_valid, seed):
    if np.intersect1d(donor, receiving).size or np.intersect1d(donor, outer_valid).size:
        raise AssertionError('Fit partition overlaps held-out rows')
    return DonorState(take_raw(RAW_TRAIN, donor), y[donor], SOURCE, seed)


def build_fold_bundle(train_idx, valid_idx, fold):
    train_x = np.empty((len(train_idx), len(FEATURE_COLS)), np.float32)
    valid_x = np.empty((len(valid_idx), len(FEATURE_COLS)), np.float32)
    test_x = np.empty((len(TEST_IDS), len(FEATURE_COLS)), np.float32)
    train_margin = np.empty(len(train_idx), np.float32)
    valid_margin = np.empty(len(valid_idx), np.float32)
    test_margin = np.empty(len(TEST_IDS), np.float32)
    inner_visits = np.zeros(len(train_idx), np.uint8)
    inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE + 3000 + fold)
    for inner_fold, (fit_local, receive_local) in enumerate(
            inner_cv.split(np.zeros(len(train_idx)), y[train_idx])):
        donor, receiving = train_idx[fit_local], train_idx[receive_local]
        state = fit_donor_state(donor, receiving, valid_idx, 7000 + 10*fold + inner_fold)
        state.transform_into(RAW_TRAIN, receiving, train_x, train_margin, receive_local)
        inner_visits[receive_local] += 1
        del state
    if not np.all(inner_visits == 1):
        raise AssertionError('Incomplete or duplicated inner OOF coverage')
    state = fit_donor_state(train_idx, valid_idx, valid_idx, 8000 + fold)
    state.transform_into(RAW_TRAIN, valid_idx, valid_x, valid_margin)
    state.transform_into(RAW_TEST, np.arange(len(TEST_IDS)), test_x, test_margin)
    del state
    for matrix in (train_x, valid_x, test_x):
        if matrix.dtype != np.float32 or matrix.shape[1] != len(FEATURE_COLS) or not np.isfinite(matrix).all():
            raise AssertionError('Feature shape, dtype, or finite check failed')
    for margin in (train_margin, valid_margin, test_margin):
        if not np.isfinite(margin).all():
            raise AssertionError('Nonfinite initial margin')
    return train_x, valid_x, test_x, train_margin, valid_margin, test_margin
