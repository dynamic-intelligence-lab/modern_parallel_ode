# Lean formalization of the continuous-time PL bounds

This directory formalizes Theorem 1 (stable case) and Theorem 2 (unstable
case) of the paper in Lean 4; "the manuscript" below means the paper. The Lean
declaration types are the authoritative record of what has been proved;
[`STATUS.md`](STATUS.md) gives the correspondence with the paper.

Verification passed on November 8, 2026: all modules compiled, and 359 project
theorem declarations (including generated lemmas) passed the transitive axiom
audit. See [`CHECK_RESULTS.txt`](CHECK_RESULTS.txt).

## Reproduce the check

The project pins Lean 4.24.0 in `lean-toolchain` and mathlib in
`lake-manifest.json`. With [elan installed](https://lean-lang.org/install/manual/),
run from this directory:

```sh
export PATH="$HOME/.elan/bin:$PATH"
lake exe cache get
./verify.sh
```

The first command downloads mathlib's compiled cache; the second builds the
proofs and audits the transitive axioms of every theorem in the `ModernODE`
namespace. Only Lean's standard `propext`, `Classical.choice`, and `Quot.sound`
are allowed. A proof placeholder or an additional mathematical axiom causes
verification to fail. No external proof oracle is used.

On a fresh clone the two commands took 5 and 9 minutes on four cores, with a
peak of 10 GB of memory; the mathlib cache takes about 5.5 GB unpacked.

`STATUS.md` gives the correspondence with the paper, the hypotheses of the
abstract `theorem1` and `theorem2`, and how `theorem1_ode'` and `theorem2_ode`
prove each of them for the IVP from their own hypotheses. The theorem names
and their full types are the authoritative record of what has been proved.

## Main results

- [`ModernODE.theorem1_ode'`](ModernODE/Theorem1ODE.lean) proves manuscript
  Theorem 1 for the IVP: zero minimum, the global PL inequality, and the chain
  `√μ⋆ ≥ c/(a(1−e^{−cT})) ≥ c/a > 0`. Theorem 1 states its main hypothesis,
  `‖Φ_s(t,τ)‖ ≤ a e^{−c(t−τ)}` for `0 ≤ τ ≤ t ≤ T` and the state-transition
  operator along every candidate, by reference to the Case 1 paragraph, which
  also gives `a ≥ 1` and `c > 0`. The "Initial value problems" paragraph gives
  `T > 0` and the state space `ℝ^D`; Lean allows any real Hilbert space of
  positive dimension. The paper leaves the other hypotheses implicit in its
  assumptions that `f` and the initial trajectory guess are sufficiently
  "nice" and that the IVP satisfies standard existence and uniqueness
  assumptions; the formalization makes them precise: `f` continuous in
  `(x,t)` and differentiable in the state, with Jacobian continuous in
  `(x,t)`; candidates in H¹ (the paper fixes `s(0) = x₀`); and a solution of
  the IVP on `[0,T]` (`hsol`). Theorem 1 itself does not state `hsol`; the
  appendix proof uses it ("the exact solution has zero residual"). It also
  follows from the decay bound, but that derivation is not formalized.
  [`STATUS.md`](STATUS.md) lists each hypothesis and where the paper has it.
  [`ModernODE.theorem1_negField`](ModernODE/NegExample.lean) discharges all
  hypotheses for `ẋ = −x`, so they are satisfiable.
- [`ModernODE.theorem2_ode`](ModernODE/Theorem2ODE.lean) proves manuscript
  Theorem 2 for the IVP: zero minimum, `√μ⋆ ≤ e λ a²/b · e^{−λT}`, and the same
  bound `√μ ≤ e λ a²/b · e^{−λT}` for every PL constant `μ`. Theorem 2 states
  `λT ≥ 2` and, by reference to the Case 2 paragraph,
  `b e^{λ(t−τ)} ≤ ‖Φ_x(t,τ)‖ ≤ a e^{λ(t−τ)}` for `0 ≤ τ ≤ t ≤ T` and the
  state-transition operator along `x`. That paragraph gives the solution `x`
  of the IVP, `a ≥ 1`, `0 < b ≤ 1` and `λ > 0` (Lean assumes `b > 0`; `b ≤ 1`
  is not needed). The "Initial value problems" paragraph gives `T > 0` and the
  state space `ℝ^D`; Lean takes it finite-dimensional of positive dimension.
  The regularity of `f` and the candidates in H¹ are implicit in the paper and
  made precise as for `theorem1_ode'`.
  [`ModernODE.theorem2_posField`](ModernODE/PosExample.lean) discharges all
  hypotheses for `ẋ = x`.
- [`ModernODE.optimalPL_tendsto_zero`](ModernODE/Corollaries.lean) proves the
  note in Theorem 2: if the hypotheses of `theorem2_ode` hold for all large
  horizons with the same `a`, `b`, `λ`, then `μ⋆ → 0` as `T → ∞`
  (`posField_optimalPL_tendsto_zero`: satisfiable for `ẋ = x`). The note
  leaves this requirement on all large horizons implicit in "for fixed `a`,
  `b`, and `λ`".
  [`ModernODE.C1Field.ofContDiff`](ModernODE/Corollaries.lean) shows that a
  field `C¹` jointly in `(x,t)` is a `C1Field`, so both IVP theorems apply to
  it.
- The abstract [`ModernODE.theorem1`](ModernODE/Theorem1.lean) and
  [`ModernODE.theorem2`](ModernODE/Theorem2.lean) take the analytic interfaces
  `StableAnalyticData` and `UnstableAnalyticData`; the two IVP theorems
  construct these interfaces from the ODE.

In summary:

> Theorems 1 and 2 have been formally verified in Lean 4 for the IVP under the
> hypotheses the paper states and under the assumptions it leaves implicit,
> made precise: `f` and its Jacobian in the state continuous in `(x,t)`,
> candidates in H¹, and, for Theorem 1, a solution of the IVP.
> Gradients are L² gradients with the extended gradient norm of the appendix;
> the state space is a real Hilbert space of positive dimension,
> finite-dimensional for Theorem 2.

`manuscript-sources.json` records the review date (2026-11-08) and the SHA-256
hashes of the draft LaTeX sources the Lean statements were compared with.
`STATUS.md` quotes and classifies the hypotheses against the arXiv version of
the paper.
`CHECK_RESULTS.txt` records the verification run. `./verify.sh` can
reproduce the build and axiom check; it does not require knowledge of Lean.
