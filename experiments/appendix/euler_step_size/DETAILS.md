# Forward-Euler step-size example: details

The short version is in [README.md](README.md). This file has every file, command and printed number.

The scalar contracting system x' = -20 x + 20 sin(2t), x(0) = 10, on [0, T] with T = 5, is discretized by forward
Euler and the discrete problem is solved by M in {50, 100, 200, 400} Jacobi sweeps. For a fixed budget M, the
total error E(M, h), the L1 distance on [0, T] between the piecewise-linear interpolant of the iterate and the
exact solution, is smallest at an intermediate step size: for large h the discretization error dominates, for
small h the M sweeps have not yet carried the initial condition through the whole horizon. The experiment compares
the step size at which E is smallest with the step-size estimates of the appendix.

In the paper this is the appendix "Jacobi-Specific Bounds for the Discretization--Parallelization Tradeoff"
(label `app:jacobi-bounds`) with its figure (label `fig:step-size`), its table of forward and backward Euler as
Jacobi step maps (label `tab:euler-jacobi`), and the subsection "Estimate of Critical Step Size" (label
`app:critical-step`). The main text refers to it from the remark on choosing the step size (label
`rem:step_size_selection`).

The paper's PDF of the figure (`figs/Euler_Error_Analysis_Plot_one_panel.pdf` in the paper's sources) was drawn
outside this repository, with Matplotlib 3.11.0, by a script that is not available. `figure_step_size.py`
recomputes its data from the definitions of the appendix and redraws it in the same layout with the Matplotlib
version of this repository (3.10.7). Every mark of the paper's figure is reproduced (see "Paper artifacts"); the
redrawn PDF is not the paper's file.

All commands are run from the repository root. NumPy and Matplotlib only; no JAX and no `parode` code.

## What the script computes

- Forward Euler step map g_k(s) = s + h f(s, t_{k-1}) on the grid t_k = k h, h = T/K. The iterate p^(M) after M
  Jacobi sweeps p_k <- g_k(p_{k-1}), k = 1..K, all states updated from the previous sweep, started from the
  constant path p^(0) = x0.
- E(M, T/K) = integral over [0, T] of |p_hat(t) - x(t)|, with p_hat the piecewise-linear interpolant of p^(M) and
  x(t) = (10 + 10/101) e^{-20t} + (100 sin 2t - 10 cos 2t)/101 the exact solution, by a composite Gauss-Legendre
  rule: 8 panels of 6 nodes on every step.
- The curves: the step sizes h = linspace(10^-4, 0.02, 500), each rounded to the nearest T/K, which gives 457
  distinct K from 250 to 50,000. The star: the smallest E over every integer K between the grid points two places
  either side of the smallest grid value (the script stops if that minimum lies at either end of this range).
- The constants c = l_x = 20, l_t = 40, vbar = 220 and d = l_x vbar + l_t = 4440 of the example;
  rho_FE(h) = e^{-ch} + e^{l_x h} - 1 - l_x h; h_asy = log(2 c^2 M T vbar / d) / (cM); h_num, the minimizer of
  max(T - Mh, 0) rho_FE(h)^M + d/(2c vbar) h over the h < T/M with rho_FE(h) < 1, located on a grid of 10^4 points
  and refined by bisection on the derivative; the dashed lines at h_mid = (h_asy + h_num)/2; the stationary point of
  psi(h) = T vbar (T - Mh) e^{-cMh} + Td/(2c) h of `app:critical-step`, by bisection.
- The figure: page 403.2 x 331.2 pt, axes box of the paper's PDF, h on a linear axis [0, 0.02] with ticks every
  0.0025, E on a log axis with the default limits, colors viridis at 0, 0.267, 0.533 and 0.8, solid curves (2.2 pt),
  dashed verticals (1.8 pt, opacity 0.85), stars with an outer radius of 7 pt, the same axis labels and legend
  entries in the same order, STIX fonts.

## Files

| file | what it is |
|---|---|
| `figure_step_size.py` | Computes the curves, the stars, the dashed lines and every number the appendix quotes about the example, prints each number next to the quoted one, and draws the figure into `figs/euler_step_size/Euler_Error_Analysis_Plot_one_panel.pdf` (`--out` for another path). |
| `paper_figure_marks.json` | The data marks of the paper's PDF of the figure: for each M the 35 to 42 vertices of the curve, the centre of the star and the position of the dashed line, in data units (h, E). How they were read is described below. Only the test reads this file; the script does not. |
| `../../../tests/test_euler_step_size.py` | The recomputed marks against `paper_figure_marks.json`, Jacobi against sequential forward Euler, and the bounds the appendix states about the example. |
| `../../../figs/euler_step_size/Euler_Error_Analysis_Plot_one_panel.pdf` | The figure as `figure_step_size.py` draws it. |

### How `paper_figure_marks.json` was read from the paper's PDF

The page content stream of the paper's PDF (sha256 recorded in the file) was decompressed and its drawing
operators interpreted, with a short standard-library Python script that is not part of the repository. The x
calibration is a linear fit to the 9 vertical grid lines and their tick labels 0.0000 to 0.0200 (largest residual
3.5x10^-18); the y calibration (log axis) a least-squares fit to the 38 horizontal grid lines at k x 10^n (largest
residual 8x10^-9 decades), whose decade is fixed by the four labelled major lines. Curves are the four 2.2 pt
stroked paths, assigned to M in legend order; the dashed lines are the four vertical 1.8 pt dashed paths; a star's
centre is the mean of its 10 vertices. The PDF stores point coordinates with 6 decimals, so a vertex's E is known
to about 2x10^-8 relative. Matplotlib leaves out of a path the points that lie almost on a line through their
neighbours, so the vertices are a subset of the points that were drawn; the last vertex of each curve is repeated
in the file. An independent extraction made during the preparation of the release, with a calibration from the 4
labelled y grid lines only, agrees with this one to 1.3x10^-8 in E.

## Paper artifacts

Levels: **A** recomputed on a CPU and compared to a stated tolerance; **B** redrawn or
re-tabulated from shipped result files; **C** needs GPUs or records that are not shipped; **D** not reproducible
from this repository.

| artifact | command | inputs | hardware and time | level |
|---|---|---|---|---|
| Figure `fig:step-size`: its data (156 curve vertices, 4 stars, 4 dashed lines) | `figure_step_size.py` | none | CPU, about 30 s | A: every mark of the paper's PDF equals the recomputation to 2.5x10^-8 relative; `tests/test_euler_step_size.py` checks 5x10^-8 |
| Figure `fig:step-size`: the PDF | `figure_step_size.py` | none | as above | a redraw of the recomputed data in the paper's layout, not the paper's file (see below) |
| Table `tab:euler-jacobi` | none | none | none | no level: the table holds formulas only |
| Numbers about the example in the text of `app:jacobi-bounds` and `app:critical-step` and in the figure's caption | `figure_step_size.py` (printed) | none | as above | A |

## Commands

```sh
python experiments/appendix/euler_step_size/figure_step_size.py
JAX_PLATFORMS=cpu python -m pytest -q tests/test_euler_step_size.py
```

`make figures` runs the first command along with the other figure scripts and compares the redrawn PDF with the
committed one. The script prints:

```
quantity                                         paper                            recomputed (M = 50, 100, 200, 400 where four values)
c = l_x (affine slope)                           20                               20  20
l_t = max |40 cos 2t|                            40                               40
vbar = max |f| on [-10, 10] x [0, T]             220                              220
d = l_x vbar + l_t                               4,440                            4440
largest h with 10|1 - 20h| + 20h <= 10           1/11                             0.0909091
rho_FE(h) < 1 for h below                        (not quoted)                     0.0465411
2 c^2 M T vbar > d                               holds                            True  True  True  True
rho_FE(h_asy) < 1                                holds                            True  True  True  True
empirical minimizer K (grid K in [250, 50000])   interior minimum                 592  1065  1956  3635
h_mid / h_min - 1, %                             2.6 to 14.2, shrinking with M    14.1354  7.36466  4.26048  2.637
E(T/round(T/h_mid)) / E_min - 1, %               at most 9                        8.42015  3.22966  1.34968  0.649947
h_asy / h_min - 1, %                             at or above, by at most 20       8.94328  5.37576  3.54658  2.51365
h_num / h_min - 1, %                             at or above, by at most 20       19.3274  9.35355  4.97439  2.76035
h_num / h_asy - 1, %                             at most 10                       9.5317  3.77486  1.3789  0.240645
c M h_asy                                        about 9 to 11                    9.20129  9.89444  10.5876  11.2807
h_asy / h_stat - 1, %                            within 1                         0.928525  0.939989  0.950492  0.960219
M h_asy / T                                      about 0.1                        0.0920129  0.0989444  0.105876  0.112807
beta M h_asy^2                                   0.85 (M = 50) to 0.16 (M = 400)  0.846637  0.489499  0.280242  0.159069
rho_FE(h_asy)^M / exp(-c M h_asy)                2.9, 1.7, 1.35, 1.18             2.91988  1.74565  1.35058  1.17955
beta h_asy / c                                   0.09, 0.05, 0.03, 0.01           0.0920129  0.0494722  0.026469  0.0141009
h_num / h_stat - 1, %                            11, 5, 2, 1                      10.5487  4.75033  2.3425  1.20317

marks of the figure:
  M =  50: star at h = T/592 = 0.00844594595, E = 0.047409504; dashed line at h_mid = 0.00963981001 (h_asy = 0.00920129054, h_num = 0.0100783295)
  M = 100: star at h = T/1065 = 0.00469483568, E = 0.0262853991; dashed line at h_mid = 0.00504059418 (h_asy = 0.00494721886, h_num = 0.00513396949)
  M = 200: star at h = T/1956 = 0.00255623722, E = 0.0142665145; dashed line at h_mid = 0.00266514529 (h_asy = 0.00264689622, h_num = 0.00268339436)
  M = 400: star at h = T/3635 = 0.00137551582, E = 0.00764417045; dashed line at h_mid = 0.00141178817 (h_asy = 0.00141009151, h_num = 0.00141348483)
wrote figs/euler_step_size/Euler_Error_Analysis_Plot_one_panel.pdf
```

h_min is the step size of the star, h = T/K_min; h_stat the stationary point of psi; "beta" is l_x^2/2. The
quantities of `app:critical-step` are evaluated at h_asy.

## The redrawn PDF against the paper's

Compared after the release preparation, with the extraction above applied to both files and with
`tools/compare_pdf.py`:

- page size, axes box and axis limits are the same; every star and every dashed line lies at the same place on
  the page to within 10^-6 pt;
- every vertex of the paper's curves is one of the 457 step sizes per M that the script draws, but the two paths
  keep different vertices: the paths of the paper's PDF run in increasing h, the script draws each curve in
  decreasing h (the order of `np.unique` on K), and Matplotlib, which leaves out of a path the points that lie
  almost on a line through their neighbours, then leaves out different points. Every vertex of either path lies
  within 0.13 pt of the other path. Drawn in increasing h, the curves have exactly the paper's vertices (the same
  step sizes in the same order, with either Matplotlib version);
- the text is the same, but the two Matplotlib versions place it differently: the origins of tick labels and axis
  labels are up to 1.9 pt apart, and the legend rows are 13.25 pt apart instead of 12.75 pt (its last row 3.8 pt
  lower), so `pdftotext -layout` breaks two lines differently;
- 4.5 % of the pixels differ at 150 dpi, from both causes. Drawn in increasing h, 4.06 % differ. With Matplotlib
  3.11.0 the text of the redraw is the same as the paper's also for `pdftotext -layout`, and 0.48 % of the pixels
  differ; 0.05 % when the curves are also drawn in increasing h. The order of the points changes no plotted value.
