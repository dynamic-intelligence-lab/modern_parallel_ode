"""Export each computed merit surface as one image, with other marks vector."""
import numpy as np
from PIL import Image
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import svg_utils as svg

def flatten_surface(fig,ax,name,assets,dpi=400):
    """Composite the mesh and optional floor contours into one image."""
    def bitmap_part(artist):
        return isinstance(artist,Poly3DCollection) or getattr(artist,'_merit_bitmap_component',False)
    fig.canvas.draw()
    position = ax.get_position().frozen()
    axes_visibility = [(other,other.get_visible()) for other in fig.axes]
    # Figure-level labels, frames and legend marks stay vector, out of the bitmap.
    text_visibility = [(text,text.get_visible()) for text in [*fig.texts,*fig.artists]]
    legend_visibility = [(legend,legend.get_visible()) for legend in fig.legends]
    artist_visibility = [(artist,artist.get_visible()) for artist in [*ax.collections,*ax.lines,*ax.texts]]
    for other,_ in axes_visibility:
        other.set_visible(other is ax)
    for text,_ in text_visibility:
        text.set_visible(False)
    for legend,_ in legend_visibility:
        legend.set_visible(False)
    for artist,_ in artist_visibility:
        artist.set_visible(bitmap_part(artist))
    ax.set_axis_off()
    width,height = fig.get_size_inches()
    bounds = Bbox.from_bounds(position.x0*width,position.y0*height,
                            position.width*width,position.height*height)
    path = assets/(name+'_surface.png')
    fig.savefig(path,dpi=dpi,transparent=True,bbox_inches=bounds,pad_inches=0)
    for artist,visible in artist_visibility:
        artist.set_visible(visible)
    for other,visible in axes_visibility:
        other.set_visible(visible)
    for text,visible in text_visibility:
        text.set_visible(visible)
    for legend,visible in legend_visibility:
        legend.set_visible(visible)
    ax.set_axis_on()
    for artist in [*ax.collections,*ax.lines]:
        if bitmap_part(artist):
            artist.remove()
    ax.patch.set_alpha(0.)
    image_ax = fig.add_axes(position,zorder=-10,label=name+'_image')
    image_ax.set_gid(name+'_image')
    pixels = np.asarray(Image.open(path).convert('RGBA'))
    image_ax.imshow(pixels,aspect='auto',interpolation='none')
    image_ax.set_axis_off()

def group_surface_image(root,name):
    parents = {child:parent for parent in root.iter() for child in parent}
    group = next(node for node in root.iter(svg.tag('g')) if node.get('id')==name)
    image_group = next(node for node in root.iter(svg.tag('g')) if node.get('id')==name+'_image')
    parents[image_group].remove(image_group)
    group.insert(0,image_group)
    assert len(list(group.iter(svg.tag('image'))))==1
