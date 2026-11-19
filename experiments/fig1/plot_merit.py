"""Opaque merit surface and its level sets projected onto an offset floor."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, LightSource, to_rgba
import style
from style import INK, BLUE, RED, axis_ink


def draw_merit(fig,g,data,position):
    stable=g<1; color=BLUE if stable else RED
    ax=fig.add_axes(position,projection='3d',computed_zorder=False)
    ax.view_init(40,-28); ax.set_proj_type('ortho')
    ax.set_box_aspect((1.1,1,1.15),zoom=1.02)
    ax.set(xlim=(-1.18,1.18),ylim=(-1,1),zlim=(-2.6,4.15))
    cmap=LinearSegmentedColormap.from_list('merit',
        ['#edfafd','#b9e1e9','#65acbf','#19788e'] if stable else
        ['#fff3df','#eaa173','#cd5a37','#8d2424'])
    A,B,raw=data['A'],data['B'],data['Z']
    Z=np.log10(1+10*raw)
    levels=np.array([.025,.07,.14,.23,.34,.43] if stable else [.4,2,8,32,128,512])
    contours=ax.contour(A,B,raw,levels=levels,zdir='z',offset=-2.6,
        colors=[color],linewidths=.70,alpha=.78,zorder=1)
    contours._merit_bitmap_component=True
    contours.set_gid(('contracting' if stable else 'chaotic')+'_floor_contours')
    rgb=cmap(Normalize(0,1.2 if stable else 4.15)(Z)); rgb[...,3]=1.
    surface=ax.plot_surface(A,B,Z,facecolors=rgb,rstride=4,cstride=4,
        shade=True,lightsource=LightSource(300,55),lw=.18,
        antialiased=True,rasterized=True,zorder=2)
    surface.set_edgecolor(to_rgba('#3a6470' if stable else '#713b2f',.32))
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.set_visible(False)
        axis._axinfo['grid']['color']=(1,1,1,0)
        axis._axinfo['tick']['inward_factor']=0
        axis._axinfo['tick']['outward_factor']=.2
    ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_zticks([0,1,2,3,4])
    ax.tick_params(labelsize=8,pad=1)
    axis_ink(ax)
    ax.scatter([0],[0],[.015],s=style.STAR_PT**2,marker='*',color='#f1bc48',
        edgecolors=INK,lw=.9,depthshade=False,zorder=9)
    return ax
