"""Draw Figure 1 of the paper, all ten subplots with their labels, from the arrays in data/.

Run: python make_figure.py
Recompute the arrays of columns A and B first, into a folder of their own:
    python make_figure.py --recompute --cache DIR
(DIR also needs the deer_*.npz of column C: run simulate_deer.py with --cache DIR, or copy them from data/).
Outputs: PDF, SVG and PNG of the figure, and the two surface images in assets/.
"""
from pathlib import Path
import argparse
import io
import os
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(HERE/'.cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import simulate_all
import style
from plot_fields import draw_field
from plot_merit import draw_merit
from export import flatten_surface, group_surface_image
from style import INK, BLUE, RED, configure, axes_style
import svg_utils as svg

ITERATIONS=[0,1,5]   # the DEER iterations of column C, top to bottom
# Layout in inches. Both rows are ROW_H tall (one frame each); add_text
# adds margins around this canvas and draws the frames.
ROW_H,GAP=4.4,.1
ROW_Y={'chaotic':.05,'contracting':.05+ROW_H+GAP}   # frame bottoms
COL_W=[ROW_H,4.25,5.9]                              # frame widths; column A is square and its field fills it
COLS=[]; x=.05                                      # frames (left, right)
for w in COL_W: COLS.append((x,x+w)); x+=w+GAP
CANVAS=(COLS[-1][1]+.05,2*ROW_H+GAP+.1)
MERIT_SIZE_IN=[4.83,3.92]
MERIT_X=COLS[1][0]-.6       # the drawn surface starts ~.9 in inside its axes
MERIT_Y={'contracting':.84,'chaotic':.4}            # above the frame bottom; the chaotic wall is tall
Z_AXIS_X,Z_LABEL_Y=3.99,1.68                         # z-axis position inside the merit axes
TRAJ_X,TRAJ_W,TRAJ_H,TRAJ_STEP=COLS[2][0]+.68,4.07,.9,1.3
ITER_LABEL_IN=.61 # a strip's right end to the centre of its 'Iteration k' label
TRAJ_TOP=.27      # frame top to the first strip; the strips and the Time label fill the frame's height
HEAD_Y=1.12       # frame top to the centre line of the headers: every title and every letter
LETTER_DX=.07     # a column's left edge to the ink of its letter, the same for A, B and C


def load(cache,kind,system):
    with np.load(cache/f'{kind}_{system}.npz') as data:
        return {k:data[k].copy() for k in data.files}


def trajectory_limits(data,selected):
    """Reference, shown iterates and the last iterate; blown-up iterates are left out."""
    ref,iterates=data['reference_coordinate'],data['coordinate_iterates']
    last=len(iterates)-1
    curves=[ref]+[iterates[min(k,last)] for k in selected+[last]]
    values=np.concatenate([c for c in curves if np.abs(c).max()<3*np.abs(ref).max()])
    lo,hi=values.min(),values.max(); pad=.11*(hi-lo)
    return lo-pad,hi+pad


def draw_trajectory(fig,data,system,step,position,selected):
    ax=fig.add_axes(position); axes_style(ax)
    color=BLUE if system=='contracting' else RED
    iterates=data['coordinate_iterates']
    lo,hi=trajectory_limits(data,selected)
    # A solver that stopped early (converged) shows its last iterate. Iterates
    # that blow up (DEER, chaotic) are clipped just outside the frame.
    y=np.clip(iterates[min(step,len(iterates)-1)],lo-10*(hi-lo),hi+10*(hi-lo))
    line,=ax.plot(data['t'],y,color=color,lw=1.65,zorder=3)
    line.set_gid(f'{system}_sweep_{step}_actual_trajectory')
    line,=ax.plot(data['reference_dense_t'],data['reference_coordinate'],
                 color='black',ls=(0,(4,3)),lw=1.,zorder=4)
    line.set_gid(f'{system}_sweep_{step}_ode_reference')
    ax.set(xlim=(0,120),ylim=(lo,hi),xticks=[0,30,60,90,120])
    ax.yaxis.set_major_locator(MaxNLocator(3)); ax.grid(axis='y',alpha=.13,lw=.5)
    ax.set_gid(f'{system}_deer_{step:03d}')
    return ax


def hide_text(fig):
    fig.canvas.draw()
    for ax in fig.axes:
        ax.xaxis.set_tick_params(which='both',labelbottom=False,labeltop=False)
        ax.yaxis.set_tick_params(which='both',labelleft=False,labelright=False)
        if hasattr(ax,'zaxis'):
            ax.zaxis.set_tick_params(which='both',labelbottom=False,labeltop=False)
    for text in fig.findobj(match=Text): text.set_visible(False)


def add_text(fig,panels,selected):
    """The labels of the published figure, drawn here instead of in Illustrator."""
    left,right,top,bottom=1.75,.05,1.55,.05   # added margins, inches
    W,H=fig.get_size_inches(); W2,H2=W+left+right,H+top+bottom
    for ax in fig.axes:  # keep every axes' size and place in inches
        x,y,w,h=ax.get_position().bounds
        ax.set_position([(x*W+left)/W2,(y*H+bottom)/H2,w*W/W2,h*H/H2])
    fig.set_size_inches(W2,H2)

    def at(x,y):  # inches on the canvas without the margins -> new figure fraction
        return (x+left)/W2,(y+bottom)/H2

    for system,y0 in ROW_Y.items():
        y1=y0+ROW_H
        color=BLUE if system=='contracting' else RED
        for x0,x1 in COLS:
            fig.add_artist(Rectangle(at(x0,y0),(x1-x0)/W2,ROW_H/H2,transform=fig.transFigure,
                                     fill=False,ec='#c9d0d5',lw=1.3))
        fig.text(*at(COLS[0][0]-.14,(y0+y1)/2),'Stable' if system=='contracting' else 'Unstable',
                 ha='right',va='center',fontsize=24,color=color)
        x0=COLS[1][0]
        fig.add_artist(Line2D([at(x0+.26,0)[0]],[at(0,y1-.28)[1]],marker='*',ms=style.STAR_PT,
                              mfc='#f1bc48',mec=INK,mew=.9,transform=fig.transFigure))
        fig.text(*at(x0+.49,y1-.28),'ODE solution',va='center',fontsize=12)
        ax=panels[system+'_merit']
        ax.set_xlabel('Trajectory Space\nDirection 1',fontsize=13,labelpad=-6)
        ax.set_ylabel('Trajectory Space\nDirection 2',fontsize=13,labelpad=-6)
        for label in [ax.xaxis.label,ax.yaxis.label]: label.set_visible(True)
        # Horizontal, just right of the z-axis.
        fig.text(*at(MERIT_X+Z_AXIS_X+.07,y0+MERIT_Y[system]+Z_LABEL_Y),r'$\mathcal{L}[s(t)]$',
                 va='center',fontsize=16)
        for row,step in enumerate(selected):
            ax=panels[f'{system}_deer_{step:03d}']
            ax.set_ylabel(r'$x_1(t)$',rotation=0,ha='right',va='center',fontsize=14,labelpad=4)
            ax.yaxis.label.set_visible(True)
            ax.text(1+ITER_LABEL_IN/TRAJ_W,.5,f'Iteration\n{step}',transform=ax.transAxes,ha='center',va='center',
                    fontsize=13,multialignment='center')
            if row==len(selected)-1:
                ax.set_xlabel(r'Time $(t)$',fontsize=16,labelpad=6); ax.xaxis.label.set_visible(True)
    headers=[('A','ODE',r'$\dot{x} = f(x,t)$',23),
             ('B','Merit Functional',
              r'$\mathcal{L}[s(t)] = \frac{1}{2}\int_0^T \|\dot{s}(t) - f(s(t),t)\|_2^2\,dt$',19),
             ('C','Parallel-in-Time\nTrajectory Optimization',r'$\delta s$',23)]
    top_frame=ROW_Y['contracting']+ROW_H
    for (x0,x1),(letter,title,formula,size) in zip(COLS,headers):
        fig.text(*at((x0+x1)/2,top_frame+.42),formula,ha='center',va='center',fontsize=size)
        # One centre line for the three titles (one or two lines) and the three letters;
        # the ink of each letter starts LETTER_DX right of its column's left edge (the bold
        # glyphs have different side bearings, so the bearing is taken off the text origin).
        fig.text(*at((x0+x1)/2,top_frame+HEAD_Y),title,ha='center',va='center',fontsize=21,linespacing=1.1)
        bearing=TextPath((0,0),letter,size=30,prop=FontProperties(weight='bold')).get_extents().x0/72
        fig.text(*at(x0+LETTER_DX-bearing,top_frame+HEAD_Y),letter,ha='left',va='center',fontsize=30,fontweight='bold')


def create_figure(cache,selected=None):
    selected=selected or ITERATIONS
    configure(plt)
    fig=plt.figure(figsize=CANVAS,facecolor='white')
    W,H=CANVAS

    def box(x,y,w,h):  # inches -> figure fraction
        return [x/W,y/H,w/W,h/H]

    panels={}; merits={}
    for system,g in [('contracting',.1),('chaotic',3.5)]:
        y0=ROW_Y[system]
        ax=draw_field(fig,g,load(cache,'field',system),
            box(COLS[0][0],y0,COLS[0][1]-COLS[0][0],ROW_H),   # the whole frame
            end_time=24 if g<1 else None)
        ax.set_gid(system+'_pc'); panels[ax.get_gid()]=ax
        ax=draw_merit(fig,g,load(cache,'merit',system),box(MERIT_X,y0+MERIT_Y[system],*MERIT_SIZE_IN))
        ax.set_gid(system+'_merit'); panels[ax.get_gid()]=ax; merits[system]=ax
        data=load(cache,'deer',system)
        for row,step in enumerate(selected):
            top=y0+ROW_H-TRAJ_TOP-TRAJ_H-TRAJ_STEP*row
            ax=draw_trajectory(fig,data,system,step,box(TRAJ_X,top,TRAJ_W,TRAJ_H),selected)
            panels[ax.get_gid()]=ax
    hide_text(fig)
    add_text(fig,panels,selected)
    return fig,panels,merits


def save_figure(cache,output,dpi=400,selected=None):
    selected=selected or ITERATIONS
    output.mkdir(parents=True,exist_ok=True)
    assets=output/'assets'; assets.mkdir(exist_ok=True)
    fig,_,merits=create_figure(cache,selected)
    for name,ax in merits.items():
        flatten_surface(fig,ax,name+'_merit',assets,dpi)
    assert not any(isinstance(c,Poly3DCollection) for ax in fig.axes for c in ax.collections)
    stem=output/'figure1_horizontal_deer_text'
    fig.savefig(stem.with_suffix('.pdf'),dpi=dpi)
    fig.savefig(stem.with_suffix('.png'),dpi=180)
    buffer=io.StringIO()
    with matplotlib.rc_context({'svg.fonttype':'none','svg.image_inline':True}):
        fig.savefig(buffer,format='svg',dpi=dpi)
    root=svg.normalize_svg(ET.fromstring(buffer.getvalue()))
    for name in merits: group_surface_image(root,name+'_merit')
    images=list(root.iter(svg.tag('image')))
    assert len(images)==2
    for image in images: image.set('style','image-rendering:auto')
    svg.write_svg(root,stem.with_suffix('.svg'))
    plt.close(fig)
    print('Complete figure: '+str(stem.with_suffix('.pdf')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute',action='store_true',help='Recompute the arrays of columns A and B (field_*, merit_*) from the model into --cache.')
    parser.add_argument('--cache',type=Path,
                        help='folder of the arrays (default data/: the shipped ones; --recompute needs one here)')
    parser.add_argument('--output',type=Path,default=HERE/'output'/'deer')
    args=parser.parse_args()
    if args.recompute and args.cache is None:
        parser.error('--recompute writes the arrays into --cache: give a folder (data/ holds the shipped arrays)')
    cache=args.cache or HERE/'data'
    simulate_all.prepare(cache,args.recompute)
    save_figure(cache,args.output)
