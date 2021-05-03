import numpy as np

from joblib import Parallel, delayed, Memory
from toolz import pipe

def _min_gap_error(X, y, si, sl=-1):
    mask = ((X > sl) & (X <= si)).all(axis=1)
    return min(((y == c) & mask).sum() for c in [-1, 1])

def _min_gap_error_init(X, y, si):
    mask = (X <= si).all(axis=1)
    return min(((y == c) & mask).sum() for c in [-1, 1])

def make_potential_boundaries(X):
    return pipe(X, np.ravel, np.unique, np.sort)

@delayed
@memory.cache
def calcG(X, y, si, s, i):
    temp=np.zeros((len(s)+1,1))
    for l, sl in enumerate(s[:i]):
        temp[l] = _min_gap_error(X, y, si, sl)
    temp[len(s)]=i
    return temp

def ontheline_quantizer(X, y, R=None, k=None):
    assert ((R is None) and (k is not None) or
            (R is not None) and (k is None))

    k = k or R - 1
    s = make_potential_boundaries(X)
    E = np.zeros((len(s), k + 1))
    A = defaultdict(lambda: defaultdict(set))

    for i, si in enumerate(s):
        E[i, 0] = _min_gap_error_init(X, y, si)

    parallel = Parallel(n_jobs=cpu_count())

    val=parallel(calcG(X,y,si,s,i) for i, si in enumerate(s))
    G=np.squeeze(np.array(val))[:, :len(s)]

    for i, si in enumerate(s):
        for b in range(1, k + 1):
            if b>0:
                if i == 0:
                    E[i, b] = 0
                    A[i][b] = set()
                else:
                    (E[i, b],
                     l_star,
                     sl_star) = min([E[l, b - 1] + G[i, l], l, sl]
                                    for l, sl in enumerate(s[:i]))
                    A[i][b] = A[l_star][b - 1].union({sl_star})
    return E, A, A[len(s) - 1][k]
