"""Visual defaults shared by all ten subplots."""
INK, MUTED, BLUE, RED = '#243646', '#75818b', '#087f91', '#bb432d'

def configure(plt):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
        'mathtext.fontset':'dejavusans', 'pdf.fonttype':42, 'text.color':INK,
        'axes.labelcolor':INK, 'axes.linewidth':.7, 'svg.hashsalt':'figure1-complete'})

# Black, 1.1 pt on the canvas (prints ~0.40 pt at 6.0 in).
AXIS_COLOR, AXIS_LW = 'black', 1.1
# The ODE-solution star of column B, pt on the canvas (prints ~7.4 pt at 6.0 in): the marker
# on each surface and the one in the legend.
STAR_PT = 20.5


def axis_ink(ax):
    """Axis lines and ticks in AXIS_COLOR, AXIS_LW wide."""
    lines = ([a.line for a in (ax.xaxis, ax.yaxis, ax.zaxis)] if hasattr(ax, 'zaxis')
             else [ax.spines['bottom'], ax.spines['left']])
    for line in lines:
        line.set_color(AXIS_COLOR)
        line.set_linewidth(AXIS_LW)
    ax.tick_params(colors=AXIS_COLOR, width=AXIS_LW)


def axes_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8,pad=2)
    axis_ink(ax)
    ax.set_axisbelow(True)
