"""DEER (Newton = Gauss-Newton on the square shooting system) for the right column.

Unknowns s_1..s_N, residual s_j - phi(s_{j-1}).
Newton step: s_j^{i+1} = phi(s_{j-1}^i) + J_j (s_{j-1}^{i+1} - s_{j-1}^i),
J_j = d phi / dx at s_{j-1}^i (exact derivative of the DOP853 map, from the tape).
DEER solves the linear recurrence with a parallel scan; here it is solved
sequentially, which gives the same iterate.
"""
from model import D, W, A, B, NONZERO, NZB, INITIAL, flow   # first: model.py sets the one-thread BLAS default
import numpy as np

T, DT, FLOW_STEP, COORDINATE = 120., .25, 1/32, 75


def initial_guess(t,x0):
    a,b=np.random.default_rng(2718).normal(size=(2,len(x0)))
    fade=np.exp(-t[:,None]/4.)
    return fade*x0+(1-fade)*(1.4*a+.8*b*np.sin(t[:,None]/17.))


def norm(a,h=DT):
    """L2 norm of the piecewise-linear interpolant of a nodal array."""
    return np.sqrt(max(0.,h/3*np.sum(a[:-1]**2+a[:-1]*a[1:]+a[1:]**2)))


def reference(g,x0):
    """Serial DOP853 rollout, then evaluate all requested times from it.

    Integer-time anchors and batched subinterval evaluations reproduce the
    numerical-reference construction of an earlier version of the figure,
    made outside this repository.
    """
    base_t=np.arange(round(T)+1,dtype=float)
    base=np.empty((len(base_t),D)); base[0]=x0
    for i,tt in enumerate(base_t[:-1]):
        base[i+1]=flow(base[i:i+1],g,np.array([tt]),duration=1.,step=FLOW_STEP)[0][0]

    def at(t):
        indices=np.minimum(np.floor(t+1e-12).astype(int),len(base_t)-1)
        offsets=t-base_t[indices]; result=base[indices].copy()
        for offset in np.unique(np.round(offsets,12)):
            if abs(offset)<1e-11: continue
            mask=np.abs(offsets-offset)<1e-10
            result[mask]=flow(base[indices[mask]],g,base_t[indices[mask]],
                duration=float(offset),step=min(FLOW_STEP,float(offset)))[0]
        return result

    t=np.arange(round(T/DT)+1)*DT
    dense_t=np.arange(round(T/.0625)+1)*.0625
    return t,at(t),dense_t,at(dense_t)


def flow_jacobian(tape, h, g):
    """Forward-mode derivative of model.flow, replaying its stage tape.

    Tangents are stored as (row, segment, column) so each stage is one GEMM.
    """
    tau = 4. if g < 1 else 1.
    n = len(tape[0][0])
    V = np.repeat(np.eye(D)[:, None, :], n, axis=1)
    for stages in tape:
        slopes = []
        for j in range(12):
            Z = V.copy()
            for a in NONZERO[j]:
                Z += h*A[j, a]*slopes[a]
            SZ = (stages[j].T[:, :, None]*Z).reshape(D, -1)
            slopes.append((g*(W @ SZ).reshape(D, n, D) - Z)/tau)
        for j in NZB:
            V += h*B[j]*slopes[j]
    return V.transpose(1, 0, 2)


def check_jacobian(g, x, t):
    """Central finite differences on two segments and three directions."""
    _, (tape, h) = flow(x, g, t, duration=DT, step=FLOW_STEP, gradient=True)
    jac = flow_jacobian(tape, h, g)
    eps = 1e-6
    for k in [0, 7, 100]:
        e = np.zeros(D); e[k] = eps
        fd = (flow(x+e, g, t, DT, FLOW_STEP)[0]-flow(x-e, g, t, DT, FLOW_STEP)[0])/(2*eps)
        np.testing.assert_allclose(jac[:, :, k], fd, atol=1e-7, rtol=1e-6)


def compute(system, cache, max_iter=480):
    g = .1 if system == 'contracting' else 3.5
    x0 = INITIAL[system].copy()
    t, ref, dense_t, dense = reference(g, x0)
    s = initial_guess(t, x0)
    check_jacobian(g, s[[3, 200]], t[[3, 200]])
    curves, errors = [], []

    def save():
        np.savez_compressed(cache/f'deer_{system}.npz', t=t, g=g, coordinate_iterates=curves,
            reference_dense_t=dense_t, reference_coordinate=dense[:, COORDINATE],
            relative_nodal_error=np.asarray(errors)/errors[0])

    for i in range(max_iter+1):
        curves.append(s[:, COORDINATE].copy()); errors.append(norm(s-ref))
        print(f'{system}: deer iter {i}  rel err {errors[-1]/errors[0]:.2e}', flush=True)
        if i % 10 == 0:
            save()
        if errors[-1] < 1e-13*errors[0] or not np.isfinite(errors[-1]) or i == max_iter:
            break
        F, (tape, h) = flow(s[:-1], g, t[:-1], duration=DT, step=FLOW_STEP, gradient=True)
        J = flow_jacobian(tape, h, g)
        new = np.empty_like(s); new[0] = x0
        for j in range(1, len(t)):
            d = new[j-1]-s[j-1]
            new[j] = F[j-1] + J[j-1] @ d
        s = new
    save()
    return np.asarray(errors)/errors[0]


if __name__ == '__main__':
    import argparse, time
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('system', choices=['contracting', 'chaotic'])
    parser.add_argument('--max-iter', type=int, default=480)
    parser.add_argument('--cache', type=Path, required=True,
                        help='existing folder to write deer_<system>.npz into (data/ holds the shipped arrays)')
    args = parser.parse_args()
    start = time.monotonic()
    compute(args.system, args.cache, args.max_iter)
    print(f'{args.system}: {time.monotonic()-start:.0f} s', flush=True)
