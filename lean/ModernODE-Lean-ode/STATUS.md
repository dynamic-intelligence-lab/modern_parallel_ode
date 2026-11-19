# Certification scope and correspondence with the manuscript

This development formalizes manuscript **Theorems 1 and 2 for the IVP
itself**, under the hypotheses the manuscript states and the assumptions it
leaves implicit, made precise; the two tables of hypotheses below say which
is which.

- `ModernODE.theorem1_ode'` in [`ModernODE/Theorem1ODE.lean`](ModernODE/Theorem1ODE.lean)
  states Theorem 1 for the concrete residual `ṡ − f(s,·)` on
  `{s ∈ H¹ : s(0) = x₀}`; see "Theorem 1 for the IVP" below. It constructs every
  field of `StableAnalyticData` from the ODE and applies the abstract
  `ModernODE.theorem1` in [`ModernODE/Theorem1.lean`](ModernODE/Theorem1.lean).
- `ModernODE.theorem2_ode` in [`ModernODE/Theorem2ODE.lean`](ModernODE/Theorem2ODE.lean)
  states Theorem 2 for the same residual; see "Theorem 2 for the IVP" below. It
  constructs every field of `UnstableAnalyticData`, and the cocycle identity of
  the transition operator, from the ODE and applies the abstract
  `ModernODE.theorem2` in [`ModernODE/Theorem2.lean`](ModernODE/Theorem2.lean).

Neither declaration assumes the desired PL bounds. The development includes
actual integral estimates, finite-dimensional operator-norm attainment, an L²
ramp construction, first-variation calculations, adjoint integration by parts,
and limit arguments.

The additional assumptions appear only in the Lean development. Run
`./verify.sh` to build all modules and audit the transitive axioms; see
[`README.md`](README.md) for setup.
The Lean declaration types are the authoritative statements of the results.

The corresponding parts of the paper ("the manuscript" below) are:

- Theorem 1 (stable case): Case 1 of the results section, with proof in the
  appendix on the continuous-time PL bounds.
- Theorem 2 (unstable case): Case 2 of the results section, with proof in the
  same appendix.
- Common trajectory, residual, and PL definitions: the paragraphs "Initial
  value problems", "Polyak–Łojasiewicz inequalities" and "Incremental
  stability" that precede the results section.

## Common meaning of the loss, first variation, and optimal constant

The formal loss is the actual squared residual norm, `residualLoss r = ‖r‖²/2`.
The first variation `meritVariation r s v` is the real derivative at ε = 0 of
`residualLoss (r (s + ε • v))`. Both analytic-data records require this
one-dimensional merit function to be differentiable at zero for every `s`
and `v`. This avoids treating Lean's default derivative value at a
nondifferentiable function as a genuine first variation.

`BoundsVariation d G` means `G ≥ 0` and `|d v| ≤ G ‖v‖` for every variation
`v`. The definition `IsPL r d μ` requires `μ ≥ 0` and

    μ ‖r(s)‖² ≤ G²

for every candidate `s` and every finite bound `G` on its first variation.
If there is no finite bound, the condition is automatic at that candidate.
This encodes the appendix's extended gradient-norm convention without
assuming an L² gradient exists everywhere. The intended manuscript
instantiation uses the trajectory L² norm, not the H¹ norm.

`Gradient.lean` proves that this definition agrees exactly with the usual
squared-gradient PL inequality when variations range over a Hilbert space
and the first variation is represented by inner product with a gradient.
It does not construct that representing gradient for the paper's residual
functional or formalize the density of its H¹ variations in L².

`optimalPL` is a supremum in ℝ. Theorem 1 explicitly supplies a nonzero-residual
point with finite first-variation bound, which proves that the set of PL
constants is bounded above. Theorem 2 derives a finite upper bound from its
perturbation family. Thus neither main theorem silently uses a real supremum
for an unbounded collection of constants. A supplied zero-residual point is
used to prove, rather than assume, that the loss infimum is zero.

The variation space `V` is abstract in `Theorem1.lean` and `Theorem2.lean`.
`ODEInstance.lean` constructs the instance for the IVP: candidates
`s = x₀ + ∫₀ᵗ u` with `u ∈ L²(0,T)` (the type `DerivCoord E T`, which is exactly
`{s ∈ H¹ : s(0) = x₀}` in derivative coordinates), the norm of a variation `v`
equal to the L² norm of the trajectory variation `∫₀ᵗ v`, and the residual
`u − f(s(·),·)` as an actual element of `L²(0,T)`. `theorem1_ode'` uses this
instance, and so does `theorem2_ode`.

## Theorem 1: exact extra assumptions and proved conclusion

The main statement assumes `a ≥ 1`, `c > 0`, `T > 0`, and the manuscript's
uniform exponential operator estimate

    ‖Φ_s(t,τ)‖ ≤ a exp(−c(t−τ)),    0 ≤ τ ≤ t ≤ T.

In addition, `StableAnalyticData` supplies the following.
These fields are hypotheses of the abstract `theorem1`; `theorem1_ode'` proves
every one of them for the IVP; it takes the zero-residual candidate from its
hypothesis `hsol` (see "Theorem 1 for the IVP" below).

| Additional data or hypothesis | Its role in the abstract `theorem1` |
| --- | --- |
| Differentiability of the merit along every variation | Makes `meritVariation` an actual directional derivative. This regularity is assumed, not derived from C¹ regularity of the vector field. |
| A time-function representation of each residual, zero outside `[0,T]`, with `‖r(s)‖² = ∫ℝ ‖residualPath(s,t)‖² dt` | Connects the abstract residual Hilbert norm to an actual integral. The support condition is explicit in `residual_support`; the intended instantiation is the zero extension of the ODE residual. |
| Integrability of the residual energy, kernel times residual norm, kernel times squared residual norm, and the latter function jointly in both times | Supplies the analytic integrability obligations for Cauchy–Schwarz and Fubini. No L² gain estimate is included among these assumptions. |
| For each candidate, an admissible correction `η` whose directional residual derivative equals `r(s)` | Expresses that the correction solves the linearized residual equation in the needed direction. `theorem1` does not construct it from `η′ = J(s,·)η + r_s`, with `η(0)=0`. |
| A time-function representation of that correction, its norm/energy identity, and its variation-of-constants integral formula using `Φ_s` | Connects the correction to the transition operator. The integral representation is assumed; its derivation from the linear ODE is not formalized. |
| Integrability of the correction's squared norm | Makes its trajectory energy finite. |
| A zero-residual candidate | Stands for an admissible exact IVP solution on the whole interval. |
| A nonzero-residual candidate with a finite bound on all first variations | Ensures the real-valued optimal PL constant is a bounded supremum. This is an additional explicit nondegeneracy/regularity assumption. |

From these hypotheses, Lean proves the correction's first-variation identity,
pointwise domination by the exponential Volterra kernel, both kernel mass
bounds, the Cauchy–Schwarz/Fubini energy estimate, and hence

    ‖η‖₂ ≤ B_T ‖r_s‖₂,    B_T = a(1−exp(−cT))/c.

It then proves the zero minimum, the global PL inequality with constant
`(c/(a(1−exp(−cT))))²`, and the full displayed chain

    √μ⋆ ≥ c/(a(1−exp(−cT))) ≥ c/a > 0.

These are composed in the main `theorem1` declaration. The norm estimate is
proved from the transition bound and integral representation; it is not an
assumption of `StableAnalyticData`.

## Theorem 1 for the IVP: `theorem1_ode'`

Every row of the table above is proved for the concrete residual of the IVP
(`ODEInstance.lean`, `TransitionODE.lean`, `Theorem1ODE.lean`):

| `StableAnalyticData` field | Proof |
| --- | --- |
| Differentiability of the merit, residual time function, integrability | `hasDerivAt_residual` (Nemytskii derivative `Dr[u](v) = v − J(s)∫₀ᵗv` from `f ∈ C¹`), `merit_differentiable_residual`, and the interface lemmas of `ODEInstance.lean` |
| Correction `η` with `Dr[s]η = r(s)` and its variation-of-constants formula | `corrDir`, `hasDerivAt_corrDir`, `voc_identity` (Fubini on the time triangle and the fundamental theorem of calculus) |
| The transition operator `Φ_s` | `exists_isTransition` (Peano–Baker series) and `isTransition_unique` (Gronwall) |
| Zero-residual candidate | the hypothesis `hsol`: the IVP has a solution on `[0,T]` |
| Nonzero-residual candidate with finite slope | `exists_finite_slope`: the exact solution up to `T−h`, then the forced IVP `ṡ = f(s,t) + φ(t)e` with a smooth bump `φ` on `(T−h,T)` (short-time Picard–Lindelöf); the slope is finite by integration by parts because this residual vanishes at `T` |

The hypotheses of `theorem1_ode'`:

| Lean | Manuscript |
| --- | --- |
| `E` a real Hilbert space with `[Nontrivial E]` | Stated elsewhere: the state space `ℝ^D` of the "Initial value problems" paragraph, which `E` generalizes; `D ≥ 1` is implicit |
| `F : C1Field E`: `f` and `J = ∂ₓf` jointly continuous | Implicit: the "Initial value problems" paragraph assumes `f` and the initial trajectory guess sufficiently "nice" for the merit functional to exist and be finite, and the "Incremental stability" paragraph uses `J = ∂f/∂x`; `C1Field` makes this precise. A field `C¹` in `(x,t)` is a `C1Field` by `C1Field.ofContDiff` |
| Candidates `u : DerivCoord E T`, that is `s = x₀ + ∫₀ᵗ u` with `u ∈ L²(0,T)`, or `{s ∈ H¹ : s(0) = x₀}`; a variation `v` is measured by the L² norm of `∫₀ᵗ v` | `s(0) = x₀`: stated elsewhere (the "Initial value problems" paragraph). The L² norm of trajectory perturbations, with perturbations in H¹ that vanish at `t = 0`: stated elsewhere (Step 3 of the proof of Theorem 1 in the appendix). Candidates in H¹: implicit in the "sufficiently nice" assumption; made precise here |
| `hsol`: some candidate has zero residual | Implicit: the "Initial value problems" paragraph speaks of "standard existence and uniqueness assumptions on the IVP", and the appendix proof uses "the exact solution has zero residual". Theorem 1 itself does not state it; uniqueness is not needed. It also follows from `hdecay` (homotopy `f − (1−θ)f(x₀,t)` gives `‖x(t) − x₀‖ ≤ a max‖f(x₀,·)‖/c`), which is not formalized |
| `ha : 1 ≤ a`, `hc : 0 < c`, `hT : 0 < T` | Stated elsewhere: `a ≥ 1` and `c > 0` in the Case 1 paragraph, with the decay estimate; `T > 0` in the "Initial value problems" paragraph |
| `hdecay`: for every candidate and every `Φ` satisfying `IsTransition` along it, `‖Φ(t,τ)‖ ≤ a e^{−c(t−τ)}`, `0 ≤ τ ≤ t ≤ T` | Stated in Theorem 1 ("Under the decay estimate"), which the Case 1 paragraph gives for every admissible trajectory and all `0 ≤ τ ≤ t ≤ T`. The paper does not define "admissible" but goes on to require "a bound on perturbation growth along all candidate trajectories"; Lean takes every candidate |

`IsTransition` is the defining system of the state-transition operator
(continuity, `Φ(τ,τ) = I`, `∂ₜΦ = J(s(t),t)Φ`); since it has exactly one
solution, `hdecay` is the manuscript's bound on `Φ_s`. `[Nontrivial E]` is needed
only because `optimalPL` is a real supremum: for `E = 0` every `μ ≥ 0` is a PL
constant and the real supremum of an unbounded set is `0`. The conclusion is
that of `theorem1`. `theorem1_negField` (`NegExample.lean`) discharges every
hypothesis for `ẋ = −x`, `x₀ = 1`, `T = 1`, `a = c = 1`.

## Theorem 2: exact extra assumptions and proved conclusion

The main statement uses a nontrivial finite-dimensional real inner-product
state space, `a ≥ 1`, `b > 0`, `λ > 0`, `T > 0`, and `λT ≥ 2`. It assumes the
upper transition bound on the full time triangle and the lower bound at the
single endpoint `(T,0)`. The paper's lower bound for every `(t,τ)` is stronger
than this endpoint condition; its restriction `b ≤ 1` is not needed by the
formal proof. The transition composition identity
`Φ(T,t) ∘ Φ(t,0) = Φ(T,0)` is explicit.

The residual map takes values in the genuine Mathlib Hilbert space
`Lp E 2 (volume.restrict (Ioc 0 T))`, abbreviated `IntervalL2 E T`.
`UnstableAnalyticData` additionally supplies the following.
These fields are hypotheses of the abstract `theorem2`; `theorem2_ode` proves
every one of them for the IVP; it takes the zero-residual candidate from its
hypotheses `u` and `hu` (see "Theorem 2 for the IVP" below).

| Additional data or hypothesis | Its role in the abstract `theorem2` |
| --- | --- |
| Differentiability of the merit along every variation | Gives the first variation its actual derivative meaning. |
| Continuity of `t ↦ Φ(t,0)v` on `[0,T]` for every state vector `v` | Allows construction of the ramp and forcing as actual L² elements. Continuity and the composition identity are supplied as transition data. |
| A linear field `A(t)` and the actual derivative identity `(Φ(t,0)v)′ = A(t)Φ(t,0)v` for every `v` and interior time `t` | Connects the transition orbits to a homogeneous differential equation. Identification of `A(t)` with the nonlinear Jacobian `J(x(t),t)`, and construction of the transition family, are not part of `theorem2`. |
| A zero-residual candidate | Represents existence of an admissible exact solution. |
| For each unit vector `v`, a continuous path `p` solving `p′ = −A(t)† p − η` on `(0,T)`, with `p(T)=0`, where `η` is the concrete ramp | Supplies an actual solution of the backward adjoint ODE, not an assumed duality identity. Existence of this solution is assumed. Lean constructs its L² representative and proves the identity `‖η‖² = ⟨p,u⟩`. |
| Sequences `s_n`, `ε_n`, and `g_n`, with `ε_n → 0`, eventually `ε_n ≠ 0`, and exact residuals `r(s_n) = ε_n p_L²` | Supplies an exact scaled-residual family using the L² representative of the same adjoint path. `theorem2` does not require `s_n` to be forced nonlinear IVP solutions or `s_n → x`. |
| Eventually `‖ε_n g_n‖` bounds all first variations at `s_n`, and `g_n → η` in L² | Supplies the analytic gradient approximation needed by the limit argument. It is a bound on the directional norm; no unproved equality with an L² gradient is inserted. `theorem2` does not derive it from the Jacobian and adjoint ODE. |

Lean chooses a unit vector attaining the final operator norm and proves that
this same direction grows throughout the interval. It defines the ramp with
`δ = 1/λ` and its forcing, proves their square integrability, and identifies
their Hilbert norms with the corresponding interval integrals. The two
exponential energy estimates are then proved by actual integration.

From the supplied homogeneous and backward differential equations, Lean
proves the ramp's differential equation and the adjoint identity by
integration by parts. The proof applies the fundamental theorem of calculus
on each side of the ramp corner and uses `η(0)=0` and `p(T)=0` to remove the
boundary terms. The resulting identity holds in the actual interval L²
space.

Lean then proves the adjoint ratio inequality and, using the supplied scaled
family, passes the PL inequality through the L² limit, bounds the supremum
of all PL constants, and derives the exact displayed conclusion

    √μ⋆ ≤ e λ a²/b · exp(−λT).

The main `theorem2` declaration composes these steps and also proves the zero
minimum. The PL upper bound, ramp energy estimates, and adjoint duality
identity are all proved; none is an assumption of `UnstableAnalyticData`.

For fixed `a`, `b`, and positive `λ`, Lean separately proves that the numerical
upper bound tends to zero as `T → ∞`. The theorem
`unstable_pl_family_tendsto_zero` also proves the conclusion for any family
`μ(T)` that is eventually nonnegative and eventually satisfies the displayed
square-root bound: `μ(T) → 0`. Applying this to the paper's optimal constants
requires the same parameters and analytic hypotheses for the relevant
horizons. For the IVP this is `optimalPL_tendsto_zero` (`Corollaries.lean`):
with the state space and `F : C1Field E` of `theorem2_ode`, `a ≥ 1`, `b > 0`,
`λ > 0`, and, for every sufficiently large `T`, a candidate with zero residual
on `[0,T]` along whose trajectory `x` every `Φ` satisfying `IsTransition`
obeys `b e^{λ(t−τ)} ≤ ‖Φ(t,τ)‖ ≤ a e^{λ(t−τ)}`, `0 ≤ τ ≤ t ≤ T`, the optimal
PL constant on `[0,T]` tends to zero (`λT ≥ 2` and `T > 0` hold for large
`T`). The paper states this conclusion in the note in Theorem 2 and leaves
the requirement on every large horizon implicit in "for fixed `a`, `b`, and
`λ`"; the other hypotheses are classified as for `theorem2_ode` in the table
below.
`posField_optimalPL_tendsto_zero` discharges the hypothesis for `ẋ = x`.

## Theorem 2 for the IVP: `theorem2_ode`

Every hypothesis of `theorem2` beyond the manuscript's is proved for the
concrete residual (`Theorem2ODE.lean`), with `A(t) = J(x(t),t)` along the
exact solution `x`:

| `theorem2` input | Proof |
| --- | --- |
| Transition operator `Φ_x`, its continuity and homogeneous equation | `exists_isTransition`, `isTransition_unique` |
| Cocycle identity `Φ(T,t)Φ(t,0) = Φ(T,0)` | `isTransition_comp` (uniqueness for the linear ODE, `linear_eq_zero_of_eq_zero`, Gronwall) |
| Differentiability of the merit, zero-residual candidate | `merit_differentiable_residual`; the hypothesis `hu` (the IVP solution) |
| Adjoint solution `−ṗ − Aᵀp = η`, `p(T) = 0`, continuous on `[0,T]` | `exists_adjoint_solution`: time reversal and the variation-of-constants path of `ODEInstance` for the transition operator of `s ↦ A(T−s)ᵀ` |
| Forced family `ṡ_ε = f(s_ε,t) + εp(t)`, `s_ε(0) = x₀`, on `[0,T]` | `exists_forced_family`: the deviation field truncated by a smooth bump is globally Lipschitz and bounded, so Picard–Lindelöf applies on all of `[0,T]`; Gronwall gives `‖s_ε − x‖∞ ≤ C\|ε\|`, so for small `ε` the truncation is inactive |
| Its residual is exactly `ε p` | `exists_candidate_of_forced` |
| First-variation bound `‖ε g_ε‖ ‖∫v‖` with `g_ε = η + (A − J(s_ε))ᵀp` | `boundsVariation_of_adjoint`: integration by parts against the tail integral `p(t) = ∫_t^T (Aᵀp + η)` (vector Fubini on the triangle, `p(T) = 0`) |
| `g_ε → η` in `L²` | `exists_adjoint_family`: uniform continuity of `J` on a compact tube around `x` |

The hypotheses of `theorem2_ode`:

| Lean | Manuscript |
| --- | --- |
| `E` finite-dimensional real inner product space with `[Nontrivial E]` | Stated elsewhere: the state space `ℝ^D` of the "Initial value problems" paragraph; `D ≥ 1` is implicit |
| `F : C1Field E` | Implicit, as for `theorem1_ode'`; the proof of Theorem 2 in the appendix uses "continuity of the Jacobian" |
| Candidates `u : DerivCoord E T` and the norm of variations | As for `theorem1_ode'` |
| `u`, `hu`: a candidate with zero residual, `x = x₀ + ∫₀ᵗ u` | Stated elsewhere: "Let `x(·)` solve" the IVP, in the Case 2 paragraph before Theorem 2 |
| `ha : 1 ≤ a`, `hb : 0 < b`, `hlam : 0 < λ`, `horizon : 2 ≤ λT`, `hT : 0 < T` | `λT ≥ 2`: stated in Theorem 2. `a ≥ 1`, `0 < b ≤ 1`, `λ > 0`: stated elsewhere, in the Case 2 paragraph (`b ≤ 1` is not needed). `T > 0`: stated elsewhere, in the "Initial value problems" paragraph |
| `hbounds`: every `Φ` satisfying `IsTransition` along `x` obeys `b e^{λ(t−τ)} ≤ ‖Φ(t,τ)‖ ≤ a e^{λ(t−τ)}`, `0 ≤ τ ≤ t ≤ T` | Stated in Theorem 2 ("Suppose … holds"), which the Case 2 paragraph gives along the solution `x` only and for all `0 ≤ τ ≤ t ≤ T` |

The conclusion contains that of `theorem2` (the zero minimum and
`√μ⋆ ≤ e λ a²/b · e^{−λT}`) and adds that every PL constant `μ` satisfies
`√μ ≤ e λ a²/b · e^{−λT}`. The addition matters because `optimalPL` is a real
supremum, which is `0` for an unbounded set: one forced candidate with nonzero
residual and finite first-variation bound shows the set of PL constants is
bounded, so `optimalPL` is the genuine supremum. `theorem2_posField` (`PosExample.lean`) discharges
every hypothesis for `ẋ = x`, `x₀ = 1`, `T = 2`, `λ = a = b = 1`.

## Coverage of the analytic proof components

| Files | What is formalized |
| --- | --- |
| `Core.lean`, `Merit.lean`, `Gradient.lean` | Actual squared-residual loss and directional derivative, zero minimum, extended-norm PL definition, real supremum handling, and equivalence with the ordinary Hilbert-gradient inequality when the derivative is represented. |
| `Stable.lean`, `StablePL.lean` | Duality-to-PL inference and exact stable constants. |
| `Schur.lean`, `Volterra.lean`, `Transition.lean` | Weighted integral Cauchy–Schwarz, Fubini/Schur energy bound, exponential kernel integrals, and operator-bound-to-integral domination. |
| `GrowingDirection.lean` | Finite-dimensional norm attainment and growth of one direction from the composition identity and transition bounds. |
| `RampEnergy.lean`, `IntervalL2.lean` | Actual ramp and forcing, exponential interval-integral energy bounds, actual L² representatives, and norm/energy identities. |
| `RampDerivative.lean` | The ramp's linearized differential equation away from its corner and almost everywhere, derived from the supplied homogeneous differential equation and used in the adjoint argument. |
| `AdjointDuality.lean` | Actual interval L² pairing and integration by parts from the homogeneous and backward ODEs, including the ramp corner and both endpoint terms. Duality is proved here; existence of the adjoint solution is proved in `Theorem2ODE.lean`. |
| `PLTransfer.lean`, `Unstable.lean`, `UnstablePL.lean` | Transfer through a scaled-residual limit, adjoint ratio, exact unstable constant, decay of the numerical bound, and convergence to zero of any eventually nonnegative PL-constant family satisfying that bound. |
| `Theorem1.lean`, `Theorem2.lean` | Conditional end-to-end compositions under the two explicit analytic-data records. |
| `ODEInstance.lean` | The IVP instance: H¹ candidates in derivative coordinates, the L² residual and its Nemytskii derivative, the correction direction and the variation-of-constants identity, and `theorem1_ode` (Theorem 1 with `Φ` and the finite-slope candidate as arguments). |
| `TransitionODE.lean` | Existence (Peano–Baker series) and uniqueness (Gronwall) of the state-transition operator. |
| `Theorem1ODE.lean` | The finite-slope candidate (forced IVP, smooth bump, integration by parts by Fubini) and `theorem1_ode'`. |
| `NegExample.lean` | All hypotheses of `theorem1_ode'` for `ẋ = −x`. |
| `Theorem2ODE.lean` | Cocycle identity, adjoint solution, forced family on `[0,T]`, first-variation bound along it, convergence `g_ε → η`, and `theorem2_ode`. |
| `PosExample.lean` | All hypotheses of `theorem2_ode` for `ẋ = x`. |
| `Corollaries.lean` | `C1Field.ofContDiff` (a field `C¹` in `(x,t)` is a `C1Field`), `optimalPL_tendsto_zero` (`μ⋆ → 0` as `T → ∞`), and its instance for `ẋ = x` at every horizon. |

## Interpretation of the paper's conventions

The two bounds are proved under the appendix's extended-gradient convention
and the existence assumptions described above. The operator-norm
lower bound in Theorem 2 is sufficient: final-time norm attainment, together
with the upper transition bound from each intermediate time to the final
time, supplies growth of one direction throughout the interval.

The paper takes candidates with fixed initial value, which the formalization
places in H¹, and its proofs speak of an L² gradient. Such a gradient need
not exist at every candidate. For example,
for `T=1`, `x₀=0`, `f(s,t)=−s`, and `s(t)=t`, integration by parts gives

    Dℒ[s](η) = 2η(1) + ∫₀¹ tη(t) dt,    η(0)=0.

Endpoint evaluation is unbounded in the L² norm. Variations that rise
linearly from zero to one over the final interval of length `1/n` have
squared L² norm `1/(3n)`, while the displayed derivative tends to 2. Thus
there is no L² gradient at this candidate. This is not a counterexample to
the appendix's convention: its extended gradient norm is infinite, and the
PL condition is automatic there. The formal definition follows that
convention. Replacing it by an H¹ gradient would change the theorem.

Existence of a solution on the whole interval also matters; C¹ regularity
alone does not prevent finite-time blow-up in a general nonlinear IVP.
The zero-residual hypotheses make this requirement visible without claiming
to establish ODE existence.

## Summary of what this development supports

A successful run of `./verify.sh` supports the following statement:

> Theorems 1 and 2 have been formally verified in Lean 4 for the IVP under the
> hypotheses the paper states and under the assumptions it leaves implicit,
> made precise: `f` and its Jacobian in the state continuous in `(x,t)`,
> candidates in H¹, and, for Theorem 1, a solution of the IVP.
> Gradients are L² gradients with the extended gradient norm of the appendix;
> the state space is a real Hilbert space of positive dimension,
> finite-dimensional for Theorem 2.

A clean build and an axiom audit establish that the Lean conclusions follow
from their stated hypotheses. This scope document and the pinned proof source
accompany that statement.
