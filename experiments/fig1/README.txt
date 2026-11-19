FIGURE 1

The paper's first figure (file figs/main_idea.pdf of the paper's source, label fig:main_idea) compares a
contracting 128-dimensional RNN (top row, "Stable") with a chaotic one (bottom row, "Unstable"). Column A
shows trajectories on a projected vector field, column B the merit functional on a plane through the
solution, column C iterates of DEER, Newton's method on the shooting system of the trajectory, at iterations
0, 1 and 5. This folder is a package of its own that computes every array of the figure and draws it. It
uses NumPy, Matplotlib and Pillow, and neither JAX nor the parode library of this repository; its DEER is
the NumPy code in simulate_deer.py.

THE FIGURE

Commands in this file are run from the repository root. The scripts find data/, inputs/ and output/ next to
themselves, so they also run from inside this folder; a relative --cache or --output is taken relative to
the current directory. The modules import each other by their flat names (import model), as scripts of one
folder do: the folder is not the package experiments.fig1, and the notebook puts it on sys.path. Its data/
holds the figure's arrays and is tracked; the top-level data/ of the repository (downloaded datasets,
ignored by Git) is another folder.

  python experiments/fig1/make_figure.py

draws the figure from the arrays in data/ in about 6 seconds on a CPU and writes
figure1_horizontal_deer_text.pdf, with .svg, .png and assets/ (the two surface images), into
experiments/fig1/output/deer/ (deer: DEER, the solver of column C). The tracked file
output/deer/figure1_horizontal_deer_text.pdf is the paper's file under another name (the same bytes). A new
run differs from it only in the creation date stored in the PDF, so Git shows the tracked file as modified
afterwards, and

  python tools/compare_pdf.py

compares its text and pixels with the committed file (it needs pdftotext and pdftoppm, poppler-utils);
git checkout -- experiments/fig1/output restores the committed bytes. Every run also writes
validation.json into the cache directory (data/ by default) and, unless MPLCONFIGDIR and XDG_CACHE_HOME are
set, the Matplotlib and fontconfig caches into experiments/fig1/.cache/. Both are ignored by Git.

The notebook figure1_paper_figure.ipynb makes the same call (ITERATIONS = [0, 1, 5]) and shows the
figure. make figures at the repository root executes it (with Jupyter, through
tools/run_figure_notebooks.slurm) and then runs tools/compare_pdf.py.

The figure reads

  inputs/initial_conditions.npz             the two initial states (model.py loads it on import)
  data/field_{contracting,chaotic}.npz      column A
  data/merit_{contracting,chaotic}.npz      column B
  data/deer_{contracting,chaotic}.npz       column C

The labels are drawn by Matplotlib (DejaVu Sans, pdf.fonttype 42). Each merit surface and its floor
contours are composited into one embedded 400-dpi image; axes, stars, vector fields and trajectory curves
remain vector. The layout constants (CANVAS, ROW_H, COLS, MERIT_X/Y, TRAJ_*, HEAD_Y, LETTER_DX) are at the
top of make_figure.py, the palette, the axis ink and the star size (STAR_PT) in style.py.

Three places where the figure or its caption says something other than what the code does:
  - Column B: the caption calls the two directions of the plane the smallest and largest singular vectors
    at the solution. The code takes the two extreme directions of a forcing-response family (see MIDDLE
    COLUMN).
  - Column C: the strips are labelled x_1(t); the curve is state coordinate 76 of 128.
  - Column C, stable row: DEER meets its stopping rule at iteration 4, so the strip labelled
    "Iteration 5" shows iterate 4.

THE DATA FILES AND HOW THEY WERE MADE

  data/field_*.npz, data/merit_*.npz (columns A and B)
    python experiments/fig1/simulate_all.py --cache DIR
    simulate_geometry.py integrates both systems over T = 30 and evaluates the merit surfaces. It also writes
    basis_<system>.npz (the reference trajectory, the two trajectory directions and their time derivatives,
    about 6 MB each), which the figure does not read and which is not shipped; the shipped field_*.npz leave
    out the array velocity (the full-state velocities), which the figure does not read either. make_figure.py
    runs this step itself for a system whose field or merit file is missing from its --cache folder, and for
    both systems with --recompute, which needs --cache (the shipped files in data/ are replaced only when
    experiments/fig1/data is given there). simulate_all.py needs --cache as well.

  data/deer_contracting.npz (column C, stable row)
    python experiments/fig1/simulate_deer.py contracting --cache DIR
    DEER from the initial guess until the relative error to the reference is below 1e-13: iterations 0 to 4,
    the last at a relative error of 8.2e-16.

  data/deer_chaotic.npz (column C, unstable row)
    python experiments/fig1/simulate_deer.py chaotic --max-iter 230 --cache DIR
    The chaotic run does not meet the 1e-13 stop: its error levels off at about 2e-9. The shipped file comes
    from a run on one core that was cancelled at iteration 236. simulate_deer.py writes its file at iteration
    0 and then every 10 iterations, so the file holds iterations 0 to 230. The first step overshoots to a
    relative error of 1.9e44; the error first falls below 1e-2 at iteration 87 and ends at 2.0e-9. The figure
    draws iterations 0, 1 and 5, and uses the last stored iterate only for the vertical limits of the strips.
    The command above stops at iteration 230 and gives an array of the same shape, but a rerun from scratch is
    not expected to be bit-identical to the shipped one: after the overshoot the iterates are set by rounding
    errors, and only iterations 0 to 5 were recomputed and compared (see below).

simulate_deer.py writes deer_<system>.npz into the folder given with --cache, which it needs (a shipped
array in data/ is replaced only when experiments/fig1/data is given there); it does not create that folder
(make it first, or the run stops with FileNotFoundError at the first save), and it rewrites the file at
iteration 0 and then every 10 iterations. make_figure.py does not compute the DEER arrays: with a new
--cache, run simulate_deer.py with the same --cache first, or copy the shipped deer_*.npz there.

Every script runs NumPy with one BLAS thread: before NumPy is loaded, model.py (imported first by the
simulate_*.py scripts) and make_figure.py set OPENBLAS_NUM_THREADS and VECLIB_MAXIMUM_THREADS (Accelerate on
macOS) to 1, unless the environment already sets them, and a value set there wins. With one thread the
chaotic iterates 0 to 5 came out equal to the shipped ones bit for bit. With several threads the result
depends on the machine: in one run with three threads iterates 1 to 5 came out different (relative error of
iterate 5: 1.3e2 in place of 3.5e1), and in runs with four threads on other nodes they came out equal. The
runs below set OPENBLAS_NUM_THREADS=1 on the command line as well.

WHAT WAS RUN FOR THIS RELEASE, AND WHAT A RECOMPUTATION COSTS

Levels as in docs/reproducing.md: (A) recomputed on a CPU in minutes and compared, (B) redrawn from files
that ship with the repository. All runs below were made on x86-64 Linux CPU nodes with 4 cores (no GPU),
with Python 3.10.19 and the package versions of requirements-lock.txt (NumPy 2.2.6, Matplotlib 3.10.7,
Pillow 11.3.0). The times are wall times of one run each. DIR stands for an empty folder outside
the repository, so that the arrays in data/ stayed as shipped.

  artifact  The paper's figure, figs/main_idea.pdf of the paper's source (label fig:main_idea)
  command   python experiments/fig1/make_figure.py, and the same with --output DIR
  inputs    data/{field,merit,deer}_*.npz, inputs/initial_conditions.npz
  cost      6 s each
  level     B. Compared with the paper's file (python tools/compare_pdf.py NEW.pdf REFERENCE.pdf): the
            extracted text is identical, and no pixel differs at 150 dpi. make figures (36 s, with the
            Lorenz figures) executes the notebook, whose PDF compares the same way.

  artifact  The arrays of columns A and B (field_*, merit_*)
  command   python experiments/fig1/simulate_all.py --cache DIR
  inputs    inputs/initial_conditions.npz
  cost      3 min 4 s
  level     A. Largest difference to the shipped arrays, relative to the largest entry of each array: field_*
            1.1e-14, merit_contracting 3.6e-15, merit_chaotic 1.5e-10. The paper's figure drawn from these
            arrays (make_figure.py --recompute --cache DIR, with the shipped deer_*.npz copied into DIR; 2 min
            59 s) has the same text as the paper's file; 0.0003 % of its pixels at 150 dpi differ, by at most
            6 of 255.

  artifact  The DEER iterates of the stable row (deer_contracting.npz)
  command   OPENBLAS_NUM_THREADS=1 python experiments/fig1/simulate_deer.py contracting --cache DIR
  inputs    inputs/initial_conditions.npz
  cost      1 min 45 s
  level     A. Every array is equal to the shipped one bit for bit (4 iterations, last error 8.2e-16).

  artifact  The DEER iterates of the unstable row (deer_chaotic.npz)
  command   OPENBLAS_NUM_THREADS=1 python experiments/fig1/simulate_deer.py chaotic --max-iter 5 --cache DIR
  inputs    inputs/initial_conditions.npz
  cost      2 min 13 s for iterations 0 to 5
  level     A for iterations 0 to 5, which include the three that the figure draws: iterates, errors and
            reference are equal to those of the shipped array bit for bit. Iterations 6 to 230 were not
            rerun: at about 26 s per iteration they need about 1.7 hours of one CPU core with
            --max-iter 230.

Everything the paper's figure draws (the last three blocks) is recomputed in about 7 minutes on one core.

Not checked: the chaotic DEER iterations after the fifth; any other platform, BLAS or NumPy version (the
bit-for-bit statements are for the machines above); the SVG and PNG files, for which no reference ships.

WHERE TO EDIT

  model.py              RNN, drive, random seeds, integrators (RK4, DOP853 flow map).
  simulate_geometry.py  T=30 reference, trajectory directions, merit values,
                        adaptive surface grid, PCA and neighboring solutions.
  simulate_all.py       Compute/cache step of columns A and B.
  simulate_deer.py      T=120 shooting problem (reference, initial guess, norm) and the DEER iterates.
  plot_fields.py        Foreground/background arrows, PC styling.
  plot_merit.py         Surface view, colors, mesh density, floor contours.
  make_figure.py        Full layout, selected iterations, trajectory panels, labels.
  style.py              Shared palette, axis styling, star size (STAR_PT).
  export.py             Single-image surface export.
  svg_utils.py          SVG grouping of the .svg output.
  figure1_paper_figure.ipynb  The notebook of the paper's figure (the same call).

After changing numerical parameters, pass --recompute with a --cache folder so old arrays of columns A and B
are not reused, and rerun simulate_deer.py for the DEER arrays. Plotting-only edits need only
python experiments/fig1/make_figure.py. The time grids and panel tick choices reproduce this particular
figure; if changing horizons or coordinate indices, update the plotting settings too.

MATHEMATICAL CONSTRUCTION

State dimension: 128. Both systems use the same W, generated as independent
standard normal entries divided by sqrt(128), with NumPy seed 42.

Contracting:
  x_dot = (-x + 0.1 W tanh(x) + u(t)) / 4.
Chaotic:
  x_dot = -x + 3.5 W tanh(x).

The contracting input is the smooth, nonperiodic two-component drive in
model.drive. Its orthonormal spatial basis uses seed 314. The two initial
states are fixed vectors stored in inputs/initial_conditions.npz; no script
of this folder generates them.
They are inputs to the IVPs, not cached solutions. Both reference solutions
are integrated afresh when recomputing.

LEFT COLUMN

Over T=30, integrate the reference and two neighboring initial conditions
with RK4 at dt=0.025. Obtain the two PCs from the centered reference trajectory.
Initial offsets are +/-1.0 along PC2 for contracting and +/-0.6 along PC1
for chaotic.
Display contracting trajectories only through t=24 to avoid wrapping around;
display chaotic trajectories through t=30. Both panels are square,
4.4 x 4.4 inches (ROW_H), drawn without axes; the field fills the frame.

Foreground arrows are the projected actual ODE velocities at the sampled
trajectory states/times. For each faint background arrow, find the nearest
projected trajectory segment, interpolate its state/time (x_a,t_a), lift the
query q to x_a + P(q-p_a), and evaluate P^T f(x_a + P(q-p_a),t_a).
This field agrees with the projected velocity on the trajectories. It is a
composite of local state/time slices, not a closed autonomous 2D reduction
of the 128D RNN. Arrows near conflicting projected crossings are omitted.
Far from every trajectory the arrows keep opacity 0.12 (BACKGROUND_MIN_OPACITY),
except where an arrow points more than 75 degrees (BACKGROUND_SEAM_DEG) away from
a grid neighbour: there the nearest trajectory point jumps, and the arrow fades out.
Lengths are normalized for visual clarity; they do not represent speeds.

MIDDLE COLUMN

The actual derivative-residual merit is
  L[s] = (1/2) integral_0^30 ||s_dot(t) - f(s(t),t)||_2^2 dt.
The admissible two-dimensional plane is
  s(t;alpha,beta) = x*(t) + alpha*v1(t) + beta*v2(t),
with v1(0)=v2(0)=0, so every point satisfies the fixed initial condition.

To construct directions, integrate V_dot=J_f(x*(t),t)V+q(t)I, V(0)=0,
where q(t)=sqrt(8/3) sin(pi*t)^2 for 0<t<1, and zero otherwise.
Diagonalize G=integral V(t)^T V(t) dt. Select its largest and smallest
eigenvectors, dividing each by the square root of its eigenvalue. Multiplying
V by these vectors gives L2-orthonormal trajectory directions. Their Hessian
curvatures at the solution are the corresponding inverse eigenvalues.
They are extreme directions in this forcing-response family, not a claim
about global extreme eigenfunctions over the entire trajectory space.
In the shipped arrays the two curvatures are 0.69 and 0.89 for the stable
system (condition number 1.3) and 3.1e-6 and 16 for the unstable one
(condition number 5.4e6).

Evaluate the nonlinear residual on this plane with trapezoidal quadrature.
The drive is additive and cancels algebraically in the residual evaluation.
A direct time-dependent RHS check is included to verify this cancellation.
The surface height is log10(1+10*L/sqrt(H1*H2)). The floor contours are level
sets of the same computed L/sqrt(H1*H2), on an offset z=-2.6 plane. Contour
levels differ by system for legibility. Neither horizontal direction is
rescaled. The chaotic grid is denser near the narrow valley.

RIGHT COLUMN

The physical time interval is T=120, with 480 intervals of length h=0.25.
The unknowns are the states s_1, ..., s_480 at the interval ends (s_0 = x0),
and the shooting residual is s_j - phi(t_j,t_(j-1),s_(j-1)), where phi is
approximated by fixed-step eighth-order DOP853 with internal step 1/32.
DEER is Newton's method on this square system (the same as Gauss-Newton):
  s_j^(i+1) = phi(s_(j-1)^i) + J_j (s_(j-1)^(i+1) - s_(j-1)^i),
with J_j the exact derivative of the DOP853 map at s_(j-1)^i, from a forward
pass over the stages. DEER solves this linear recurrence with a parallel scan;
simulate_deer.py solves it sequentially, which gives the same iterate. The
run stops when the relative L2 error to the reference falls below 1e-13, when
it is not finite, or at --max-iter. A finite-difference check of J_j runs
before the first iteration.
The same smooth bad-guess construction (seed 2718) is used for both systems.
Colored curves show actual state coordinate 76 (Python index 75); black
dashed curves show an independently integrated ODE reference (a serial
DOP853 rollout). The T=30 merit planes illustrate conditioning and are not a
visualization of a descent path on that plane.

REPRODUCIBILITY AND VALIDATION

The preparation step writes validation.json, which records the numerical
checks of columns A and B, into the cache directory. Background-field
agreement with trajectory velocities is checked during plotting; the export
requires two embedded images and no vector surface meshes.

Chaotic trajectories amplify floating-point differences; BLAS/platform and
NumPy changes can produce small differences, especially over T=120. Fixed
seeds, initial states, eigenvector sign conventions, and integrator choices
are supplied. Exact image-byte equality across platforms is not promised.
No script depends on files outside this directory or on an absolute path;
the notebook finds the folder from the repository root.

The DOP853 coefficient table in model.py is attributed to the SciPy/Hairer
implementation in its source comment. It is included to avoid a SciPy
runtime dependency.
