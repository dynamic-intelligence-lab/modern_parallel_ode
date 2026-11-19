"""PC trajectories and trajectory-conditioned RNN background arrows."""
import numpy as np
import matplotlib
from model import H, rhs
from style import INK, MUTED, BLUE, RED
ARROW_LENGTH_IN=.12
BACKGROUND_LENGTH_IN=.095
# Background arrows: spacing on the page (in), and the distance from the nearest
# trajectory over which they fade, as a fraction of the panel's size (geometric mean
# of its sides), so a larger panel keeps the same look.
BACKGROUND_SPACING_IN=1/6
BACKGROUND_FADE=.177
# Far from every trajectory the arrows keep this opacity (.23 on a trajectory), so the
# field fills the panel; 0 restores the fade to nothing. An arrow that points more than
# BACKGROUND_SEAM_DEG away from a grid neighbour sits where the nearest trajectory point
# jumps (a seam of the construction): it keeps the fade to nothing instead.
BACKGROUND_MIN_OPACITY=.12
BACKGROUND_SEAM_DEG=75

def field_limits(d):
    bounds=d['projected'].reshape(-1,2)
    lo=bounds.min(axis=0);hi=bounds.max(axis=0)
    pad=np.array([.12,.20])*(hi-lo)
    return lo-pad,hi+pad

def arrow_samples(d,g,axes_inches):
    """Space samples along the rendered curves while retaining actual times."""
    lo,hi=field_limits(d)
    indices=[];curves=[]
    for k in range(3):
        if g<1 and k>0:
            # Later neighbors coincide visually with the reference. Draw their
            # arrows in the resolved transient, rather than stacking duplicates.
            ids=np.rint(np.array([.6,2.2,4.2,6.3,8.7])/H).astype(int)
        else:
            step=np.diff(d['projected'][k],axis=0)/(hi-lo)
            arc=np.r_[0,np.cumsum(np.linalg.norm(step,axis=1))]
            fractions=np.linspace(.04,.96,13 if k==0 else 10)
            ids=np.searchsorted(arc,fractions*arc[-1])
            ids=np.clip(ids,2,len(arc)-3)
        indices.extend(ids);curves.extend([k]*len(ids))
    indices=np.asarray(indices);curves=np.asarray(curves)
    velocities=d['projected_velocity'][curves,indices]
    # Normalize lengths only for display, preserving the exact tangent direction
    # under the axes' affine coordinate transform. No speed comparison is shown.
    display_speed=np.linalg.norm(velocities/(hi-lo)*np.asarray(axes_inches),axis=1)
    multiplier=np.divide(ARROW_LENGTH_IN,display_speed,
        out=np.zeros_like(display_speed),where=display_speed>1e-14)
    return dict(indices=indices,curves=curves,times=d['t'][indices],
        positions=d['projected'][curves,indices],velocities=velocities,
        displacements=velocities*multiplier[:,None],arrow_length_inches=ARROW_LENGTH_IN)

def conditioned_velocity(d,g,coords,axes_inches,check_ambiguity=True):
    """Evaluate the RNN in PC planes anchored to the nearest trajectory point.

    For query q, interpolate an anchor (x_a,t_a) on the nearest projected
    trajectory segment and evaluate P.T f(x_a + P(q-p_a), t_a). This agrees
    with the projected ODE velocity at trajectory samples. It is a composite
    of local state/time slices, not an autonomous two-dimensional reduction.
    """
    coords=np.asarray(coords)
    lo,hi=field_limits(d)
    scale=np.asarray(axes_inches)/(hi-lo)
    p=d['projected'];ncurves,nt,_=p.shape;nseg=nt-1
    starts=p[:,:-1].reshape(-1,2)
    ends=p[:,1:].reshape(-1,2)
    segment=(ends-starts)*scale
    norm2=np.einsum('ij,ij->i',segment,segment)
    midtimes=np.tile(.5*(d['t'][:-1]+d['t'][1:]),ncurves)
    best_ids=[];best_weights=[];best_distances=[]
    alternatives=[];alternative_weights=[];alternative_distances=[]
    for first in range(0,len(coords),64):
        query=coords[first:first+64]
        offset=(query[:,None]-starts[None])*scale
        weight=np.clip(np.divide(np.einsum('bsi,si->bs',offset,segment),norm2,
                      out=np.zeros(offset.shape[:2]),where=norm2>1e-24),0.,1.)
        distance=np.linalg.norm(offset-weight[...,None]*segment,axis=-1)
        row=np.arange(len(query));idx=distance.argmin(axis=1)
        best_ids.append(idx);best_weights.append(weight[row,idx])
        best_distances.append(distance[row,idx])
        if check_ambiguity:
            # Consider nearby competing trajectories and temporally separate
            # branches of the same trajectory, excluding adjacent segments.
            per_curve=distance.reshape(len(query),ncurves,nseg).argmin(axis=2)
            per_curve+=np.arange(ncurves)[None]*nseg
            far=np.abs(midtimes[None]-midtimes[idx,None])>2.
            distant=np.where(far,distance,np.inf).argmin(axis=1)
            candidates=np.column_stack((per_curve,distant))
            alternatives.append(candidates)
            alternative_weights.append(np.take_along_axis(weight,candidates,axis=1))
            alternative_distances.append(np.take_along_axis(distance,candidates,axis=1))

    indices=np.concatenate(best_ids);weights=np.concatenate(best_weights)
    distances=np.concatenate(best_distances)

    def evaluate(query,idx,w):
        curve=idx//nseg;i=idx%nseg
        anchor=(1-w[:,None])*d['states'][curve,i]+w[:,None]*d['states'][curve,i+1]
        anchor_pc=(1-w[:,None])*p[curve,i]+w[:,None]*p[curve,i+1]
        times=(1-w)*d['t'][i]+w*d['t'][i+1]
        lifted=anchor+(query-anchor_pc)@d['projection'].T
        velocity=rhs(lifted,g,times)@d['projection']
        projection_error=np.max(np.abs((lifted-d['center'])@d['projection']-query))
        assert projection_error<1e-10
        return velocity,times,curve,i

    velocity,times,curves,segments=evaluate(coords,indices,weights)
    ambiguous=np.zeros(len(coords),dtype=bool)
    if check_ambiguity:
        candidates=np.concatenate(alternatives)
        candidate_weights=np.concatenate(alternative_weights)
        candidate_distances=np.concatenate(alternative_distances)
        alt,_,_,_=evaluate(np.repeat(coords,candidates.shape[1],axis=0),
                          candidates.ravel(),candidate_weights.ravel())
        alt=alt.reshape(len(coords),-1,2)*scale
        v=velocity*scale
        cosine=np.divide(np.einsum('bci,bi->bc',alt,v),
                         np.linalg.norm(alt,axis=-1)*np.linalg.norm(v,axis=-1)[:,None],
                         out=np.ones(candidate_distances.shape),
                         where=(np.linalg.norm(alt,axis=-1)*np.linalg.norm(v,axis=-1)[:,None])>1e-20)
        conflicts=(candidate_distances<.085)&(distances[:,None]<.085)&(cosine<np.cos(np.deg2rad(40)))
        ambiguous=conflicts.any(axis=1)
    return dict(coords=coords,velocity=velocity,anchor_times=times,
                anchor_curves=curves,anchor_segments=segments,
                anchor_weights=weights,distance_inches=distances,ambiguous=ambiguous)

def background_samples(d,g,axes_inches):
    lo,hi=field_limits(d)
    # Keep approximately square spacing on the page as panel proportions
    # change. Arrow lengths are likewise normalized in physical page units.
    columns=round(axes_inches[0]/BACKGROUND_SPACING_IN)
    rows=max(3,round((columns+1)*axes_inches[1]/axes_inches[0])-1)
    xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],columns+2)[1:-1],
                      np.linspace(lo[1],hi[1],rows+2)[1:-1])
    coords=np.column_stack((xx.ravel(),yy.ravel()))
    result=conditioned_velocity(d,g,coords,axes_inches)
    speed=np.linalg.norm(result['velocity']/(hi-lo)*np.asarray(axes_inches),axis=1)
    multiplier=np.divide(BACKGROUND_LENGTH_IN,speed,
                         out=np.zeros_like(speed),where=speed>1e-12)
    result['displacements']=result['velocity']*multiplier[:,None]
    fade=BACKGROUND_FADE*np.sqrt(np.prod(axes_inches))
    shown=result['velocity']/(hi-lo)*np.asarray(axes_inches)   # directions as drawn on the page
    angle=np.arctan2(shown[:,1],shown[:,0]).reshape(rows,columns)
    turn=np.zeros((rows,columns))                              # largest angle to a grid neighbour
    da=np.abs(np.angle(np.exp(1j*np.diff(angle,axis=0)))); turn[:-1]=np.maximum(turn[:-1],da); turn[1:]=np.maximum(turn[1:],da)
    da=np.abs(np.angle(np.exp(1j*np.diff(angle,axis=1)))); turn[:,:-1]=np.maximum(turn[:,:-1],da); turn[:,1:]=np.maximum(turn[:,1:],da)
    floor=np.where(turn.ravel()<np.deg2rad(BACKGROUND_SEAM_DEG),BACKGROUND_MIN_OPACITY,0.)
    result['opacity']=np.maximum(.23*np.exp(-(result['distance_inches']/fade)**2),floor)
    result['visible']=(~result['ambiguous'])&(speed>1e-12)&(result['opacity']>.035)

    # Verify this same background construction against the actual velocities
    # on each plotted trajectory, rather than only checking the foreground.
    ids=np.unique(np.linspace(2,len(d['t'])-3,41).round().astype(int))
    query=d['projected'][:,ids].reshape(-1,2)
    check=conditioned_velocity(d,g,query,axes_inches,check_ambiguity=False)
    exact=d['projected_velocity'][:,ids].reshape(-1,2)
    relative=np.linalg.norm(check['velocity']-exact,axis=1)/np.maximum(np.linalg.norm(exact,axis=1),1e-14)
    assert relative.max()<1e-7
    return result

def draw_field(fig,g,d,position,end_time=None):
    """Column A in the axes at position (figure fraction), without a frame: the field fills the panel.
    d: the arrays of field_<system>.npz."""
    if end_time is not None:
        keep=d['t']<=end_time+1e-10
        d={**d,'t':d['t'][keep]}
        for key in ['states','velocity','projected','projected_velocity','separation']:
            if key in d:d[key]=d[key][:,keep]
    axes_inches=np.asarray(position[2:])*fig.get_size_inches()
    a=arrow_samples(d,g,axes_inches);col=BLUE if g<1 else RED
    ax=fig.add_axes(position)
    p=d['projected']
    q=background_samples(d,g,axes_inches);visible=q['visible']
    xy=q['coords'][visible];vv=q['displacements'][visible]
    colors=np.tile(matplotlib.colors.to_rgba(col),(visible.sum(),1))
    colors[:,3]=q['opacity'][visible]
    artist=ax.quiver(xy[:,0],xy[:,1],vv[:,0],vv[:,1],angles='xy',
        scale_units='xy',scale=1,color=colors,width=.0022,headwidth=4.3,
        headlength=5.1,pivot='mid',zorder=.5)
    artist.set_gid(('contracting' if g<1 else 'chaotic')+'_background_field')
    for k in [1,2]:
        ax.plot(p[k,:,0],p[k,:,1],c=col,alpha=.48,lw=1.05,zorder=2)
    ax.plot(p[0,:,0],p[0,:,1],c=col,lw=1.6,zorder=3)
    for k in range(3):
        mask=a['curves']==k;xy=a['positions'][mask];vv=a['displacements'][mask]
        ax.quiver(xy[:,0],xy[:,1],vv[:,0],vv[:,1],
            angles='xy',scale_units='xy',scale=1,color=INK if k==0 else col,
            alpha=.95 if k==0 else .75,width=.004,headwidth=5.0,
            headlength=5.5,pivot='mid',zorder=4)
        ax.scatter(p[k,0,0],p[k,0,1],s=22,facecolor='white',edgecolor=col,lw=.9,zorder=5)
    if g<1:
        for t in [0,4,8]:   # dotted lines joining the three trajectories at t = 0, 4, 8
            i=round(t/H)
            ax.plot(p[[2,0,1],i,0],p[[2,0,1],i,1],c=MUTED,lw=.7,
                ls=(0,(1.5,2)),zorder=1)
    lo,hi=field_limits(d)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    ax.set_axis_off()   # no axis lines or ticks: the field fills the panel
    return ax
