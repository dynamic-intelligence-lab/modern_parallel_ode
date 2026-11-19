"""Shared RNN model and fixed-step eighth-order numerical flow map.

The two supplied initial vectors are fixed experimental inputs; no cached
trajectories or rendered artwork are required to run this package.
"""
from pathlib import Path
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
D, H = 128, .025
W = np.random.default_rng(42).normal(size=(D,D))/np.sqrt(D)
PDRIVE = np.linalg.qr(np.random.default_rng(314).normal(size=(D,2)))[0]
with np.load(Path(__file__).resolve().parent/'inputs/initial_conditions.npz') as data:
    INITIAL = {name:data[name].copy() for name in data.files}

def drive(t):
    t=np.asarray(t)
    phase=2*np.pi*(t/30+.00005*t*t)+.4
    envelope=1+.25*np.sin(2*np.pi*t/79+.2)
    controls=np.stack((.6*np.cos(2*np.pi*t/(30*np.sqrt(2))+.7),
                       envelope*np.sin(phase)),axis=-1)
    return 1.8*np.sqrt(D)*(controls@PDRIVE.T)

def autonomous_rhs(x,g):
    return -x+g*(np.tanh(x)@W.T)

def rhs(x,g,t):
    return (autonomous_rhs(x,g)+drive(t))/4 if g<1 else autonomous_rhs(x,g)

def rk4(x,g,t,h=H):
    a=rhs(x,g,t)
    b=rhs(x+h*a/2,g,t+h/2)
    c=rhs(x+h*b/2,g,t+h/2)
    d=rhs(x+h*c,g,t+h)
    return x+h*(a+2*b+2*c+d)/6

def pulse(t):
    return np.sqrt(8/3)*np.sin(np.pi*t)**2 if 0<t<1 else 0.

# Numerical DOP853 tableau (nodes C and coefficients A; row 12 of A is the weight vector B), copied
# from SciPy v1.15.3, scipy/integrate/_ivp/dop853_coefficients.py:
# https://github.com/scipy/scipy/blob/v1.15.3/scipy/integrate/_ivp/dop853_coefficients.py
#     Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
#     BSD 3-Clause License
# SciPy describes its DOP853 as a Python implementation of the DOP853 algorithm originally written
# in Fortran (Hairer, Norsett, Wanner).
# Licence text: THIRD_PARTY_NOTICES.md at the repository root, section "SciPy".
# Only the twelve main stages are needed; no SciPy dependency is required.
C = np.array([0., .0526001519587677318785587544488,
    .0789002279381515978178381316732, .118350341907227396726757197510,
    .281649658092772603273242802490, 1/3, .25, 4/13,
    .651282051282051282051282051282, .6, 6/7, 1.])
A = np.zeros((13, 12))
TABLE = [
    (1,0,.0526001519587677318785587544488),
    (2,0,.0197250569845378994544595329183),(2,1,.0591751709536136983633785987549),
    (3,0,.0295875854768068491816892993775),(3,2,.0887627564304205475450678981324),
    (4,0,.241365134159266685502369798665),(4,2,-.884549479328286085344864962717),(4,3,.924834003261792003115737966543),
    (5,0,.037037037037037037037037037037),(5,3,.170828608729473871279604482173),(5,4,.125467687566822425016691814123),
    (6,0,.037109375),(6,3,.170252211019544039314978060272),(6,4,.0602165389804559606850219397283),(6,5,-.017578125),
    (7,0,.0370920001185047927108779319836),(7,3,.170383925712239993810214054705),(7,4,.107262030446373284651809199168),(7,5,-.0153194377486244017527936158236),(7,6,.00827378916381402288758473766002),
    (8,0,.624110958716075717114429577812),(8,3,-3.36089262944694129406857109825),(8,4,-.868219346841726006818189891453),(8,5,27.5920996994467083049415600797),(8,6,20.1540675504778934086186788979),(8,7,-43.4898841810699588477366255144),
    (9,0,.477662536438264365890433908527),(9,3,-2.48811461997166764192642586468),(9,4,-.590290826836842996371446475743),(9,5,21.2300514481811942347288949897),(9,6,15.2792336328824235832596922938),(9,7,-33.2882109689848629194453265587),(9,8,-.020331201708508626135822928593),
    (10,0,-.93714243008598732571704021658),(10,3,5.18637242884406370830023853209),(10,4,1.09143734899672957818500254654),(10,5,-8.14978701074692612513997267357),(10,6,-18.5200656599969598641566180701),(10,7,22.7394870993505042818970056734),(10,8,2.49360555267965238987089396762),(10,9,-3.0467644718982195003823669022),
    (11,0,2.27331014751653820792359768449),(11,3,-10.5344954667372501984066689879),(11,4,-2.00087205822486249909675718444),(11,5,-17.9589318631187989172765950534),(11,6,27.9488845294199600508499808837),(11,7,-2.85899827713502369474065508674),(11,8,-8.87285693353062954433549289258),(11,9,12.3605671757943030647266201528),(11,10,.643392746015763530355970484046),
    (12,0,.0542937341165687622380535766363),(12,5,4.45031289275240888144113950566),(12,6,1.89151789931450038304281599044),(12,7,-5.8012039600105847814672114227),(12,8,.31116436695781989440891606237),(12,9,-.152160949662516078556178806805),(12,10,.201365400804030348374776537501),(12,11,.0447106157277725905176885569043),
]
for row, col, val in TABLE:
    A[row, col] = val
B = A[12].copy()
NONZERO = [np.flatnonzero(A[i, :i]) for i in range(12)]
NZB = np.flatnonzero(B)
np.testing.assert_allclose(A[:12].sum(axis=1), C, atol=1e-14)
np.testing.assert_allclose(B.sum(), 1., atol=1e-14)


def flow(x, g, t, duration=.25, step=1/32, gradient=False):
    """Batched numerical phi(t+duration,t,x), with an optional derivative tape."""
    n = round(duration / step)
    h = duration / n
    tape = []
    for k in range(n):
        stages, slopes = [], []
        tt = t + k*h
        for j in range(12):
            z = x.copy()
            for a in NONZERO[j]:
                z += h*A[j, a]*slopes[a]
            th = np.tanh(z)
            tau = 4. if g < 1 else 1.
            dz = (-z + g*(th @ W.T))/tau
            if g < 1:
                dz += drive(tt + C[j]*h)/tau
            slopes.append(dz)
            if gradient:
                stages.append(1-th*th)
        x = x.copy()
        for j in NZB:
            x += h*B[j]*slopes[j]
        if gradient:
            tape.append(stages)
    return x, (tape, h)
