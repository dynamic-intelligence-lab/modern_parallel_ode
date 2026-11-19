"""Recompute both left columns: RNN/variational trajectories and true merits.

The two plotted trajectory directions are normalized extreme responses to
an early forcing pulse. See README.txt for the mathematical construction.
No rendered images or previously computed trajectories are read here.
"""
import numpy as np
from model import D, H, W, INITIAL, rhs, autonomous_rhs, rk4, pulse


def trajectory_basis(g, x0, horizon=30., h=H):
    t=np.arange(round(horizon/h)+1)*h
    x=np.zeros((len(t),D)); x[0]=x0
    v=np.zeros((len(t),D,D)); eye=np.eye(D)
    tau=4. if g<1 else 1.

    def fun(xx,vv,tt):
        th=np.tanh(xx)
        jac=(-eye+g*W*(1-th*th)[None,:])/tau
        return rhs(xx,g,tt), jac@vv+pulse(tt)*eye

    for i,tt in enumerate(t[:-1]):
        a,av=fun(x[i],v[i],tt)
        b,bv=fun(x[i]+h/2*a,v[i]+h/2*av,tt+h/2)
        c,cv=fun(x[i]+h/2*b,v[i]+h/2*bv,tt+h/2)
        d,dv=fun(x[i]+h*c,v[i]+h*cv,tt+h)
        # Each system keeps the arithmetic order of the code that first computed it
        # (an earlier version of the figure, made outside this repository).
        if g<1:
            x[i+1]=x[i]+h*(a+2*b+2*c+d)/6
            v[i+1]=v[i]+h*(av+2*bv+2*cv+dv)/6
        else:
            x[i+1]=x[i]+h/6*(a+2*b+2*c+d)
            v[i+1]=v[i]+h/6*(av+2*bv+2*cv+dv)
    weights=np.full(len(t),h); weights[[0,-1]]*=.5
    gram=np.einsum('t,tik,til->kl',weights,v,v,optimize=True)
    ev,evec=np.linalg.eigh(gram)
    C=evec[:,[-1,0]]/np.sqrt(ev[[-1,0]])
    # Eigenvector signs are arbitrary. This convention keeps the left/right
    # orientation of the earlier version of the figure across eigensolver implementations.
    desired=np.array([1.,-1.]) if g<1 else np.array([-1.,-1.])
    actual=np.sign(C[np.argmax(abs(C),axis=0),np.arange(2)])
    C*=desired*actual
    directions=v@C
    jv=(-directions+g*np.einsum('ij,tj,tja->tia',W,
        1-np.tanh(x)**2,directions,optimize=True))/tau
    derivatives=jv+np.array([pulse(tt) for tt in t])[:,None,None]*C[None]
    hessian=1/ev[[-1,0]]
    result=dict(t=t,x=x,dirs=directions,derivs=derivatives,weights=weights,
                ev=ev,g=g,hessian=hessian,scale=np.sqrt(np.prod(hessian)))
    check=np.einsum('t,tia,tib->ab',weights,directions,directions,optimize=True)
    np.testing.assert_allclose(check,np.eye(2),atol=2e-8)
    np.testing.assert_array_equal(directions[0],np.zeros((D,2)))
    return result


def merit_values(d, coefficients):
    """L[x* + alpha*v1 + beta*v2] = (1/2) integral ||s' - f(s,t)||^2.

    x* is numerically integrated and x*'=f(x*,t). The additive drive cancels
    from the residual exactly. Trapezoidal quadrature uses step .025.
    """
    coefficients=np.asarray(coefficients)
    x,v,dv,w=(d[k] for k in ['x','dirs','derivs','weights'])
    g=float(d['g']); tau=4. if g<1 else 1.
    fx=autonomous_rhs(x,g)/tau
    ans=np.empty(len(coefficients))
    for i in range(0,len(coefficients),32):
        c=coefficients[i:i+32]
        delta=np.einsum('pa,tia->pti',c,v)
        residual=np.einsum('pa,tia->pti',c,dv)+fx[None]-autonomous_rhs(x[None]+delta,g)/tau
        ans[i:i+32]=.5*np.einsum('t,pti,pti->p',w,residual,residual)
    return ans


def merit_grid(d):
    """Sample the display grid of the earlier version of the figure; refine near the chaotic valley."""
    if float(d['g'])<1:
        A,B=np.meshgrid(np.linspace(-1,1,81),np.linspace(-1,1,81))
    else:
        aa=np.linspace(-1,1,101); ends=[]
        # Dense central grid spans normalized merit <= 1.25. Extend to the
        # full [-1,1]^2 box without rescaling either trajectory coordinate.
        for sign in [-1.,1.]:
            lo,hi=np.zeros_like(aa),np.ones_like(aa)
            edge=merit_values(d,np.column_stack((aa,sign*hi)))/d['scale']
            clipped=edge<=1.25
            for _ in range(22):
                mid=(lo+hi)/2
                z=merit_values(d,np.column_stack((aa,sign*mid)))/d['scale']
                below=z<1.25
                lo=np.where(below,mid,lo); hi=np.where(below,hi,mid)
            ends.append(sign*np.where(clipped,1.,(lo+hi)/2))
        center=ends[0][None]+np.linspace(0,1,121)[:,None]*(ends[1]-ends[0])[None]
        q=np.linspace(0,1,31)
        lower=-1+q[:-1,None]*(center[0]+1)[None]
        upper=center[-1][None]+q[1:,None]*(1-center[-1])[None]
        B=np.concatenate((lower,center,upper))
        A=np.broadcast_to(aa[None],B.shape).copy()
    Z=merit_values(d,np.column_stack((A.ravel(),B.ravel()))).reshape(A.shape)/d['scale']
    return dict(A=A,B=B,Z=Z,scale=d['scale'],hessian=d['hessian'])


def pc_trajectories(d):
    x=d['x']; t=d['t']; g=float(d['g']); center=x.mean(axis=0)
    _,singular,vt=np.linalg.svd(x-center,full_matrices=False)
    projection=vt[:2].T
    for j in range(2):
        if projection[np.argmax(abs(projection[:,j])),j]<0: projection[:,j]*=-1
    if g<1 and (x[-1]-x[0])@projection[:,0]<0: projection[:,0]*=-1
    offset=1. if g<1 else .6
    perturbation_pc=1 if g<1 else 0
    states=np.empty((3,len(x),D)); states[0]=x
    states[1,0]=x[0]+offset*projection[:,perturbation_pc]
    states[2,0]=x[0]-offset*projection[:,perturbation_pc]
    for i,tt in enumerate(t[:-1]):
        states[1:,i+1]=rk4(states[1:,i],g,tt,t[i+1]-tt if g<1 else H)
    velocity=rhs(states,g,t)
    separation=np.linalg.norm(states[1:]-states[0],axis=-1)
    if g<1:
        rate=(1-g*np.linalg.norm(W,2))/4
        assert np.all(separation<=offset*np.exp(-rate*t)[None]+1e-8)
    return dict(t=t,states=states,velocity=velocity,center=center,projection=projection,
        projected=(states-center)@projection,projected_velocity=velocity@projection,
        separation=separation,initial_perturbation_norm=offset,
        projected_variance_fraction=np.sum(singular[:2]**2)/np.sum(singular**2))


def compute(system, cache):
    g=.1 if system=='contracting' else 3.5
    print(system+': integrating ODE and variational equations (T=30)',flush=True)
    d=trajectory_basis(g,INITIAL[system])
    np.savez_compressed(cache/f'basis_{system}.npz',**d)
    print(system+': evaluating the actual merit surface',flush=True)
    grid=merit_grid(d)
    np.savez_compressed(cache/f'merit_{system}.npz',**grid)
    fields=pc_trajectories(d)
    np.savez_compressed(cache/f'field_{system}.npz',**fields)
    # Check the cancellation used above against direct time-dependent f.
    c=np.array([.4,-.7]); delta=d['dirs']@c
    residual=d['derivs']@c+rhs(d['x'],g,d['t'])-rhs(d['x']+delta,g,d['t'])
    direct=.5*np.einsum('t,ti,ti->',d['weights'],residual,residual)
    np.testing.assert_allclose(direct,merit_values(d,c[None])[0],rtol=1e-11)
    return dict(hessian=d['hessian'].tolist(),condition_number=float(d['hessian'][1]/d['hessian'][0]),
                merit_grid_shape=list(grid['Z'].shape),direct_merit_check_passed=True)
