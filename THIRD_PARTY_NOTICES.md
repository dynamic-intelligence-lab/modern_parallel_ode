# Third-party notices

This repository contains code that was ported or adapted from other projects, numeric tables taken
from another project, and code that reads data published by others. This file lists each source,
what was taken from it and where it sits, and the notices that its licence asks us to keep. The
repository's own code is under the BSD 3-Clause License of the file `LICENSE`.

Every copyright line and licence text below was copied from the source itself: its LICENSE file
(for data, its Readme) in a clone of the source repository, in an installed distribution, or in the
project's public repository. Each section names which one was read. Public repositories were read
on 2026-11-12. Nothing in this file is legal advice.

## Overview

| # | Source | Licence | Files in this repository | What was taken |
|---|--------|---------|--------------------------|----------------|
| 1 | ELCD | MIT | `experiments/elcd/model.py`, `flow_gatherfree.py`, `train/metrics.py`, `train/data.py`; `tests/reference_flow.py`, `tests/reference_systems.py` | port from PyTorch to JAX; re-implemented helpers |
| 2 | manifold-flow | MIT | `experiments/elcd/flow_gatherfree.py`, `tests/reference_flow.py` | port (reached through ELCD) |
| 3 | Neural Spline Flows | MIT | `experiments/elcd/flow_gatherfree.py`, `tests/reference_flow.py` | port (reached through manifold-flow) |
| 4 | ELK | BSD 3-Clause | `parode/elk_baseline.py` | re-implementation |
| 5 | dynamax | MIT | `parode/elk_baseline.py` | adaptation of one operator |
| 6 | parallel-mcmc | BSD 3-Clause | `parode/deer_map.py` | adaptation |
| 7 | DEER | BSD 3-Clause | `parode/deer_map.py` | adaptation (reached through parallel-mcmc) |
| 8 | SciPy | BSD 3-Clause | `experiments/fig1/model.py` | numeric tables |
| 9 | parallel-ode | no licence stated by the source | `parode/explicit_parallel.py`, `parode/sequential.py`, `parode/sequential_integrators.py`, helper functions of `parode/quasi_parallel.py`, `newton_step` of `parode/exp_parallel.py` | origin of the solver core |
| 10 | predictability_enables_parallelization | MIT | `tests/ode_problems.py` (one function), `parode/jacobi.py` | copy with small changes; adaptation of one function |
| 11 | LASA dataset, ELCD pendulum data, EigenWorms data | see section 11 | not in this repository | data read at run time |
| 12 | Mathlib and other dependencies | Apache-2.0 (Mathlib) | `lean/ModernODE-Lean-ode` depends on it | dependency only, nothing vendored |
| 13 | micro_deer | MIT | `experiments/appendix/robot_control/neural/implicit_deer.py` | adaptation |

---

## 1. ELCD (Extended Linearized Contracting Dynamics)

- **Project:** the code of "Learning Neural Contracting Dynamics: Extended Linearization and Global
  Guarantees" (S. Jaffe, A. Davydov, D. Lapsekili, A. Singh, F. Bullo, NeurIPS 2024),
  <https://github.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics>
- **Source read:** a clone at commit `87c171f` (2024-12-06): `LICENSE`, `README.md`, `models/elcd.py`,
  `utils.py`, `data.py` and the package `manifold_flow/`.
- **What this repository took, and where:**

  | File | Kind | Taken from |
  |---|---|---|
  | `experiments/elcd/model.py` | port, PyTorch to JAX | `models/elcd.py`: the latent field of class `ELCD` (`build_model`, `init_weights`, `get_A_matrix`, `forward`) |
  | `experiments/elcd/flow_gatherfree.py`, `tests/reference_flow.py` | port, PyTorch to JAX | the normalizing flow that `utils.generate_mf_transform` builds from the repository's `manifold_flow/` package (sections 2 and 3 say where that package comes from) |
  | `tests/reference_systems.py` | re-implementation of one formula | `make_data_ode`: the data-space field of `ELCD_Transform.forward` (`models/elcd.py`), written as one linear solve |
  | `experiments/elcd/train/metrics.py` | re-implementation | `chamfer` (the distance of `utils.dtwd_np`) and `truncate` (the stopping rule of `utils.forward_step_until_converge`) |
  | `experiments/elcd/train/data.py` | re-implementation | `load_lasa_raw`, `normalize` and `load_pendulum` follow `data.lasa_to_torch` (start 15), `data.NormalisedTrajData` (no mean subtraction) and `data.pendulum_to_torch` |
  | not in this repository | data read at run time | `data/pendulum/n-{2,4,8}.npz`, see section 11.2 |

  Other files under `experiments/elcd/` name the ELCD paper or repository (its protocol, its
  hyper-parameters, its data) without holding code from it, as far as their own statements and a
  line-overlap scan against the clone show: `train/pendulum.py`, `train/losses.py`,
  `train/train_elcd.py`.
- **Copyright line:** `Copyright (c) 2024 Sean Jaffe`
- **Licence:** MIT License
- **What was changed:** the PyTorch modules were rewritten as pure, unbatched JAX functions over
  NamedTuple parameter trees, synthetic initialisers were added, and `flow_gatherfree.py` uses dense
  0/1 matrices in place of the index gathers of the direct port (`tests/reference_flow.py`, the
  reference of the test of the flow's values); no file of the source is copied verbatim.
- **Licence text** (the `LICENSE` file of the source, verbatim):

```text
MIT License

Copyright (c) 2024 Sean Jaffe

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 2. manifold-flow

- **Project:** the code of "Flows for simultaneous manifold learning and density estimation"
  (J. Brehmer and K. Cranmer, NeurIPS 2020), <https://github.com/johannbrehmer/manifold-flow>
- **Source read:** the public repository, branch `public` at commit `b201b38`: `LICENSE.md`,
  `README.md` and ten source files. No local copy existed.
- **Why it is listed:** the ELCD repository (section 1) contains a package `manifold_flow/` that
  carries no licence file and does not name manifold-flow as its origin. A comparison with the
  manifold-flow repository shows that it is a copy with small modifications. Counting the lines
  that a `diff` marks on either side:
  `transforms/splines/rational_quadratic.py` and `nn/resnet.py` are identical;
  `transforms/lu.py` differs in 6 lines, `transforms/permutations.py` in 10, `transforms/base.py` in
  17, `transforms/linear.py` in 21, `utils/various.py` in 30, `transforms/projections.py` in 36 and
  `transforms/coupling.py` in 57; ELCD's `manifold_flow/vector_transforms.py` differs in 2 lines from
  manifold-flow's `experiments/architectures/vector_transforms.py`.
- **What this repository took, and where:** `experiments/elcd/flow_gatherfree.py` and
  `tests/reference_flow.py` are a port to JAX of `create_vector_transform`,
  `RandomPermutation`, `LULinear`, `PiecewiseRationalQuadraticCouplingTransform`, `ResidualNet`,
  `unconstrained_rational_quadratic_spline`, `rational_quadratic_spline`, `various.searchsorted` and
  `various.create_alternating_binary_mask`, in the one configuration that ELCD uses.
- **Copyright line:** `Copyright (c) 2020 Johann Brehmer and Kyle Cranmer`
- **Licence:** MIT License
- **What was changed:** the same as in section 1: a port from PyTorch to JAX of one configuration
  (LU linear layers, rational-quadratic coupling with linear tails); no file is copied verbatim.
- **Licence text** (`LICENSE.md` of the source, verbatim):

```text
MIT License

Copyright (c) 2020 Johann Brehmer and Kyle Cranmer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Neural Spline Flows (nsf)

- **Project:** the code of "Neural Spline Flows" (C. Durkan, A. Bekasov, I. Murray,
  G. Papamakarios, NeurIPS 2019), <https://github.com/bayesiains/nsf>
- **Source read:** the public repository, branch `master` at commit `1dcc0f2`: `LICENSE.md` and six
  source files. No local copy existed.
- **Why it is listed:** the README of manifold-flow (section 2) states: "The code is largely based on
  the excellent [Neural Spline Flow code base](https://github.com/bayesiains/nsf) by C. Durkan,
  A. Bekasov, I. Murray, and G. Papamakarios". The classes and functions that `flow_gatherfree.py`
  and `tests/reference_flow.py` port exist under the same names in nsf: `LULinear` (`nde/transforms/lu.py`), `RandomPermutation`
  (`nde/transforms/permutations.py`), `PiecewiseRationalQuadraticCouplingTransform`
  (`nde/transforms/coupling.py`), `unconstrained_rational_quadratic_spline`
  (`nde/transforms/splines/rational_quadratic.py`), `ResidualNet` (`nn/resnet.py`), `searchsorted`
  and `create_alternating_binary_mask` (`utils/torchutils.py`).
- **What this repository took, and where:** `experiments/elcd/flow_gatherfree.py` and
  `tests/reference_flow.py`, through manifold-flow and ELCD.
- **Copyright line:** `Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios`
- **Licence:** MIT License
- **What was changed:** the code reached this repository through two intermediate copies and was
  then ported from PyTorch to JAX; no file is copied verbatim.
- **Licence text** (`LICENSE.md` of the source, verbatim):

```text
MIT License

Copyright (c) 2019 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 4. ELK

- **Project:** the code of "Towards Scalable and Stable Parallelization of Nonlinear RNNs"
  (X. Gonzalez, A. Warrington, J. T. H. Smith, S. W. Linderman, NeurIPS 2024),
  <https://github.com/lindermanlab/elk>
- **Source read:** a local copy without version-control metadata: `LICENSE`, `README.md`,
  `elk/algs/elk.py`, `elk/utils/parallel_kalman_scalar.py`. Its `LICENSE` and `elk/algs/elk.py` are
  byte-identical to the `main` branch of the public repository.
- **What this repository took, and where:** `parode/elk_baseline.py` re-implements the ELK
  update of `elk_alg` (`elk/algs/elk.py`): the linear-Gaussian state-space model on the linearized
  recurrence and the filtered means as the new iterate. Its diagonal filter (`_diag_elements`,
  `_diag_combine`) follows the scalar filter of `elk/utils/parallel_kalman_scalar.py`, which states
  that it is "adapted from dynamax/linear_gaussian_ssm/parallel_inference.py" (section 5). Other
  files only import this module.
- **Copyright line:** `Copyright (c) 2024, Linderman Lab`
- **Licence:** BSD 3-Clause License
- **What was changed:** where the source calls dynamax's general filter and its own scalar filter,
  this file writes the filter elements of the ELK model in closed form, vectorises the scalar filter
  over coordinates and attaches the update to this repository's Euler recurrence with a
  residual-based stop; no file is copied verbatim.
- **Licence text** (the `LICENSE` file of the source, verbatim):

```text
BSD 3-Clause License

Copyright (c) 2024, Linderman Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 5. dynamax

- **Project:** dynamax, <https://github.com/probml/dynamax>
- **Source read:** the installed distribution dynamax 1.0.2: its `LICENSE`, its metadata
  (`License: MIT License`) and `dynamax/linear_gaussian_ssm/parallel_inference.py`. The `LICENSE` is
  byte-identical to the one at tag `0.1.5` (the version ELK pins) and on `main` of the public
  repository.
- **What this repository took, and where:** in `parode/elk_baseline.py`, `_dense_combine`
  follows the associative operator of `lgssm_filter` and `_dense_elements` follows
  `_initialize_filtering_messages` of `parallel_inference.py`.
- **Copyright line:** `Copyright (c) 2022 Probabilistic machine learning`
- **Licence:** MIT License
- **What was changed:** the log-normaliser and the symmetrisation were dropped, the operator is
  batched by broadcasting, and the elements are written in closed form for the ELK model.
- **Licence text** (the `LICENSE` file of the distribution, verbatim):

```text
MIT License

Copyright (c) 2022 Probabilistic machine learning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 6. parallel-mcmc

- **Project:** the code of "Parallelizing MCMC Across the Sequence Length" (D. M. Zoltowski, S. Wu,
  X. Gonzalez, L. Kozachkov, S. W. Linderman), <https://github.com/lindermanlab/parallel-mcmc>
- **Source read:** a clone at commit `5e5b637` (2026-03-02): `LICENSE`, `README.md`, `src/qdeer.py`,
  `src/windowed_qdeer.py`, `src/deer.py`.
- **What this repository took, and where:** `parode/deer_map.py`. `quasi_diag_estimator`,
  `_par_solve` and the Newton loop of `deer_map_solve` are adapted from `src/qdeer.py`
  (`quasi_diag_estimator`, `diagonal_matmul_recursive`, `diagonal_seq1d_inv_lin`,
  `diagonal_deer_iteration_helper`); `deer_map_solve_full` and `_par_solve_full` are the
  dense-Jacobian form of the same loop, as in `src/deer.py`.
- **Copyright line:** `Copyright (c) 2025, Linderman Lab`
- **Licence:** BSD 3-Clause License
- **What was changed:** the generic helper with its shifter and inverse-operator callbacks was
  flattened into two solvers that reuse this repository's scan primitives.
- **Licence text** (the `LICENSE` file of the source, verbatim):

```text
BSD 3-Clause License

Copyright (c) 2025, Linderman Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 7. DEER

- **Project:** the code of "Parallelizing non-linear sequential models over the sequence length"
  (Lim et al., ICLR 2024), <https://github.com/machine-discovery/deer>
- **Source read:** a clone at commit `409140f` (2025-06-23): `LICENSE`, `README.md`, and
  `deer/deer_iter.py` and `deer/seq1d.py` at commit `17b0b62`, where the `LICENSE` is the same.
- **Why it is listed:** the three parallel-mcmc files of section 6 each open with: "Code adapted from
  the original DEER codebase by Lim et al. (2024): https://github.com/machine-discovery/deer
  Based on commit: 17b0b625d3413cb3251418980fb78916e5dacfaa (1/18/24) Copyright (c) 2023, Machine
  Discovery Ltd Licensed under the BSD 3-Clause License (see LICENSE file for details)."
- **What this repository took, and where:** `parode/deer_map.py`, through parallel-mcmc: the shape of
  the Newton loop of `deer_iteration_helper` (`deer/deer_iter.py`: the tolerance by dtype, the
  clipping of the iterate, the `while_loop`) and the linear solve by an associative scan of
  `matmul_recursive` (`deer/seq1d.py`).
- **Copyright line:** `Copyright (c) 2023, Machine Discovery Ltd`
- **Licence:** BSD 3-Clause License
- **What was changed:** see section 6; the code reached this repository in the modified form of
  parallel-mcmc.
- **Licence text** (the `LICENSE` file of the source, verbatim):

```text
BSD 3-Clause License

Copyright (c) 2023, Machine Discovery Ltd

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 8. SciPy

- **Project:** SciPy, <https://github.com/scipy/scipy>
- **Source read:** the installed distribution scipy 1.15.3: `LICENSE.txt` of its metadata,
  `scipy/integrate/_ivp/dop853_coefficients.py` and `scipy/integrate/_ivp/rk.py`. The coefficient
  file is byte-identical to tag `v1.15.3` of the public repository, and the licence text below is
  identical to `LICENSE.txt` at that tag.
- **What this repository took, and where:** numeric tables only, in
  `experiments/fig1/model.py` (the arrays under the comment that names SciPy): the twelve nodes `C` and the 58 non-zero
  entries of the coefficient matrix `A` of the DOP853 method (row 12 of `A` is the weight vector `B`).
  The arrays that the file builds are equal, entry by entry and to the last bit, to `C[:12]`,
  `A[:13, :12]` and `B` of SciPy 1.15.3. The error-estimator and dense-output coefficients were not
  taken, and no SciPy code was: the integration loop (`flow`) is this repository's own.
- **Copyright line:** `Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.`
- **Licence:** BSD 3-Clause License (the three-clause text below; the metadata of the distribution
  carries the classifier `License :: OSI Approved :: BSD License`).
- **What was changed:** the non-zero entries of `A` were re-typed as a sparse list of (row, column,
  value) triples, and three of the nodes `C` are written as the fractions 1/3, 4/13 and 6/7 where
  SciPy has decimal literals.
- **Note:** SciPy describes its DOP853 as "a Python implementation of "DOP853" algorithm originally
  written in Fortran", cites Hairer, Norsett and Wanner and the page of the original Fortran code,
  and says that "the algorithmic core and coefficients are the same". `LICENSES_bundled.txt` of
  SciPy v1.15.3 has no entry for it. The original Fortran distribution and its terms were not read.
- **Licence text** (the opening section of `LICENSE.txt`, verbatim; the rest of that file concerns
  libraries bundled into SciPy's binary wheels, none of which is in this repository):

```text
Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## 9. parallel-ode

The solver library `parode` of this repository started from parallel-ode and keeps its package name.

- **Project:** parallel-ode by Casian Iacob, <https://github.com/casiacob/parallel-ode>, the code of
  C. Iacob, H. Razavi and S. Särkkä, "A parallel-in-time Newton's method-based ODE solver",
  arXiv:2511.01465 (2025).
- **Source read:** a clone at commit `f6c4f27` (2025-11-07): `README.md`, `setup.py` and the five
  files of `parode/`; and the public repository on 2026-11-12 (branch `main` at commit `7831274`).
- **What this repository holds from it, and where:**

  | File | From the source | Extent |
  |---|---|---|
  | `parode/explicit_parallel.py` | `parode/explicit_parallel.py`: `rk4_step`, `euler_step`, `step_functions`, `affine_terms`, `linear_terms`, `combine_fc`, `par_scan`, `par_init`, `newton_step`, `explicit_par_integrate` | 60 of the 84 lines of the source file are unchanged in the 170-line file |
  | `parode/sequential.py` | `parode/sequential.py`: `seq_integrate` | 12 of 17 source lines unchanged (33 lines now) |
  | `parode/sequential_integrators.py` | `parode/sequential_integrators.py`: `euler`, `rk4`, `integrators` | 11 of 27 source lines unchanged (34 lines now) |
  | `parode/quasi_parallel.py` | helper functions taken from `parode/explicit_parallel.py` and modified for a diagonal Jacobian: `combine_fc`, `par_scan`, `par_init`, `newton_step`, and the `while_loop` driver of `quasi_par_integrate`; it imports `euler_step` and `affine_terms` from this repository's `parode/explicit_parallel.py` | the helpers; the rest of the file is new |
  | `parode/exp_parallel.py` | `newton_step`, with the same origin; it imports `par_scan` and `par_init` from this repository's `parode/explicit_parallel.py`. `linear_and_affine_terms` shares its name and role with a function of the source's `parode/implicit_parallel.py`; its body is new. | `newton_step`; the rest of the file is new |

  The line counts of this repository's files include their 3-line origin header. The source's files
  `parode/implicit_parallel.py` and `parode/parareal.py` and its folder `examples/` are not in this repository
  (this repository's `examples/own_ode.py` is new).
  Other files here (`parode/deer_map.py`, `parode/elk_baseline.py`, `parode/implicit.py` and others)
  import the functions above,
  so the solvers of this repository depend on them. `experiments/elcd/solvers.py` defines
  `_par_init`, a rewrite of `par_init` with allocations that keep the dtype.
- **What was changed:** a time argument and an input function were threaded through every step, the
  Newton loop stops on a tolerance, and the LASER path (the `laser` argument) was added.
- **Licence:** the source repository contains no licence file and no copyright line (as read on
  2026-11-12), so none is reproduced here. The code is used with acknowledgment of its author.

## 10. predictability_enables_parallelization

- **Project:** the code of "Predictability Enables Parallelization of Nonlinear State Space Models"
  (X. Gonzalez, L. Kozachkov, D. M. Zoltowski, K. L. Clarkson, S. W. Linderman, NeurIPS 2025),
  <https://github.com/lindermanlab/predictability_enables_parallelization>
- **Source read:** a local copy without version-control metadata: `LICENSE`, `src/lle.py` and
  `src/deer.py`, each byte-identical to `main` at commit `9bf10fe` of the public repository.
- **What this repository took, and where:** `tests/ode_problems.py`:
  `estimate_lle_from_vector` is a copy of the function of the same name in `src/lle.py`; 23 of its
  27 non-blank lines are verbatim. The other functions of the file are this repository's own.
  `tests/test_laser.py` imports the function. `parode/jacobi.py`: `jacobi_alg` is adapted from
  `deer_alg` in `src/deer.py`: the step function with its NaN reset, the stopping test and the
  single-length body of its `while_loop`, and the layout of its return value; 55 of the 80 non-blank
  lines below the file's header occur verbatim in `src/deer.py`.
  `experiments/appendix/contracting_ssm/cell_solvers.py` calls it for the Jacobi solver.
- **Copyright line:** `Copyright (c) 2025 Linderman Lab`
- **Licence:** MIT License
- **What was changed:** in `estimate_lle_from_vector`, the argument `Ts` and its branch were replaced
  by a step `dt`, by which the result is divided to give the exponent per unit of time, and the
  docstring was shortened. In `jacobi_alg`, the Newton step of `deer_alg` (Jacobians, damping,
  clipping and the associative scan) was replaced by a Jacobi sweep, in which every state becomes `f`
  of the previous state of the current iterate; the merit function is an argument (`merit_fxn`) and is
  evaluated on the new iterate in the loop body; the defaults are `reset=True` and `tol=5e-4`; the
  scan over a fixed number of iterations (`full_trace`), the evaluation at several lengths (`Ts`) and
  the Lyapunov-exponent estimates were removed, and the function returns seven values.
- **Licence text** (the `LICENSE` file of the source, verbatim):

```text
MIT License

Copyright (c) 2025 Linderman Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 11. Data that is downloaded, not shipped

None of the datasets below is part of this repository. The code reads them from folders that the
user fills (environment variables `ELCD_LASA_DIR`, `ELCD_PENDULUM_DIR`, `EIGENWORMS_DIR`);
`experiments/elcd/download_data.sh` fetches the first two, `experiments/appendix/contracting_ssm/download_data.sh`
the third.

### 11.1 LASA handwriting dataset

- **Source:** <https://bitbucket.org/khansari/lasahandwritingdataset> ("LASA Handwriting Dataset,
  version 2.0"); a clone at commit `38304f7` (2015-03-24) was read: its `Readme.txt`; its
  `DataSet/` folder holds 30 `.mat` files.
- **Read by:** `experiments/elcd/train/data.py` (`load_lasa_raw` reads `<shape>.mat` from the
  `DataSet` folder; default `data/lasa/DataSet`).
- **Copyright notice of the Readme** (lines 1 to 4 of `Readme.txt`, verbatim):

```text
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%    Copyright (c) 2010 S. Mohammad Khansari-Zadeh, LASA Lab, EPFL,   %%%
%%%          CH-1015 Lausanne, Switzerland, http://lasa.epfl.ch         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

- **Terms and requested citation of the Readme** (lines 43 to 49 of `Readme.txt`, word for word;
  in the file three of these lines end in a space, which is not reproduced here):

```text
This library of motion is free for non-commercial academic use. The library
must not be modified or distributed without prior permission of the authors.
Please acknowledge the authors in any academic publications that have made
use of this library or part of it. Please use this BibTex for reference:

 S. M. Khansari-Zadeh and A. Billard, "Learning Stable Non-Linear Dynamical
 Systems with Gaussian Mixture Models", IEEE Transaction on Robotics, 2011.
```

- The Readme names no standard licence; the paragraph above is the whole of its terms.
- **What is and is not in this repository:** no file of the dataset and no array of demonstrations.
  `experiments/elcd/download_data.sh` clones the authors' repository unchanged. The two shipped data
  files of the ELCD figures, `experiments/elcd/schematic/schematic_data.npz` and
  `figs/elcd/appendix/seed_trajectories.pkl`, hold outputs of trained models, the
  equilibrium point, the segment boundaries and the normalization constants; the scripts that draw
  from them (`experiments/elcd/schematic/make_schematic_C.py`,
  `experiments/elcd/figures/figure_appendix.py`) rebuild the demonstrations from the
  downloaded dataset. The figures `figs/elcd/elcd_figure_traj_left.pdf` and
  `figs/elcd/appendix/seed_trajectories_chain8z_fp32.pdf` draw demonstrations of the chained
  task as curves, as the paper does.

### 11.2 Pendulum data of the ELCD repository

- **Source:** `data/pendulum/n-2.npz`, `n-4.npz`, `n-8.npz` of the ELCD repository (section 1). The
  repository's `LICENSE` (MIT, `Copyright (c) 2024 Sean Jaffe`) is its only licence statement; it has
  no separate statement for the data files.
- **Read by:** `experiments/elcd/train/data.py` (`load_pendulum`, systems `pend:<n>`) and
  `experiments/elcd/train/pendulum.py` (`load_paper`, `fit_paper`), from `data/pendulum` by
  default. The constants `PAPER_MODEL` in `pendulum.py` are the result of a fit to these files.

### 11.3 EigenWorms data of the DEER repository

- **Source:** `experiments/04_rnn_eigenworms/neuralrde_split/eigenworms_{train,val,test}.pkl` of the
  DEER repository (section 7) at commit `409140f`: the EigenWorms recordings (Yemini et al., 2013) in
  the train, validation and test split of Morrill et al. (2021), as pickled PyTorch tensors. The
  repository's `LICENSE` (BSD 3-Clause, `Copyright (c) 2023, Machine Discovery Ltd`, section 7) is its
  only licence statement; it has no separate statement for these files. The terms of the original
  recordings were not reviewed for this file.
- **Read by:** `experiments/appendix/contracting_ssm/eigenworms_data.py` (`load_eigenworms`), from
  `data/eigenworms` by default. `experiments/appendix/contracting_ssm/download_data.sh` downloads the three
  files at that commit and checks their SHA-256.

---

## 12. Dependencies that are not vendored

- **Mathlib.** `lean/ModernODE-Lean-ode` requires mathlib4 at revision
  `f897ebcf72cd16f89ab4577d0c826cd14afaafc7` (`lakefile.toml`, `lake-manifest.json`); Lake downloads
  it at build time, and no Mathlib file is in this repository. At that revision Mathlib is under the
  Apache License 2.0 (its `LICENSE`; its files say "Released under Apache 2.0 license as described
  in the file LICENSE") and has no NOTICE file.
- **Other Lake packages** named in `lake-manifest.json` (plausible, LeanSearchClient, importGraph,
  proofwidgets, aesop, Qq, batteries, Cli) are dependencies of Mathlib, downloaded the same way and
  not vendored. In a Lake checkout of the pinned revisions their `LICENSE` files open as Apache
  License 2.0, except Cli, which is MIT.
- **Python packages** that the code imports (jax, numpy, scipy, matplotlib, pandas, equinox, optax,
  Pillow, nbformat, IPython, pytest, and optionally torch and wandb) are
  installed by the user and are not redistributed here. Their licences were not reviewed for this
  file, apart from SciPy (section 8).

---

## 13. micro_deer

- **Project:** micro_deer, <https://github.com/lindermanlab/micro_deer>
- **Source read:** a clone: `src/micro_deer/algs/micro_deer.py` at commit `cc2a82c` (2026-03-07), the
  version `implicit_deer.py` was written from (the file did not change until commit `09e0224`, 2026-07-05):
  `deer_implicit` and its `_deer_implicit_fwd` and `_deer_implicit_bwd`, lines 260 to 312, added in
  commit `0a39450`, 2026-03-06; `micro_deer_alg`, lines 137 to 166. `LICENSE` and `pyproject.toml`
  (`license = {text = "MIT"}`) at the same commit; the `LICENSE` file is the same at the later commit
  `b534dc1` (2026-07-06).
- **What this repository took, and where:** `experiments/appendix/robot_control/neural/implicit_deer.py` adapts
  `deer_implicit`: a `jax.custom_vjp` whose forward pass is the Newton loop of the parallel solver (the
  `while_loop` driver of `micro_deer_alg` with its NaN guard
  `X_new = jnp.where(jnp.isnan(X_new), initial_guess, X_new)`) and whose backward pass applies the
  implicit function theorem at the converged trajectory: the adjoint recursion
  lambda_t = g_t + J_{t+1}^T lambda_{t+1} over the step Jacobians as a reverse-time `lax.scan`, the
  gradient with respect to the initial state through the Jacobian of the first step, and a zero gradient
  for the initial guess. 12 of the 133 lines of `implicit_deer.py` that are longer than 12 characters
  also occur in `micro_deer.py`: two imports, seven lines of the `while_loop` driver, whose other lines
  it shares with `explicit_par_integrate` of parallel-ode (section 9), and three lines of the backward
  pass.
- **Copyright line:** `Copyright (c) 2026 Linderman Lab`
- **Licence:** MIT License
- **What was changed:** the step maps are the exponential and explicit-Euler steps of `parode` instead
  of a discrete map `f(state, driver)`; the gradient goes to a pytree of parameters, by one VJP of the
  time-batched step map, instead of to the drivers through the per-step Jacobians with respect to the
  driver; a NaN update size is replaced by 1e3 so that the loop goes on. Upstream rewrote `deer_implicit`
  in commit `09e0224` (2026-07-05; gradient for the parameters only, adjoint by a parallel scan);
  `implicit_deer.py` does not follow that version.
- **Licence text** (the `LICENSE` file of the source, verbatim):

```text
MIT License

Copyright (c) 2026 Linderman Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
