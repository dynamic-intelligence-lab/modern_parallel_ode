import ModernODE.UnstablePL
import ModernODE.Merit
import ModernODE.GrowingDirection
import ModernODE.AdjointDuality

/-!
# Theorem 2 with an explicit adjoint/nonlinear interface

The transition operator, its growing direction, the ramp, its forcing,
and their L² energies are concrete. The adjoint identity is proved from
a supplied solution of the backward ODE. The existence of that solution
and the nonlinear exact-residual family are hypotheses of this abstract
statement; `Theorem2ODE` constructs them from the ODE (`theorem2_ode`). In
particular, no PL upper bound or exponential energy estimate is assumed.
-/

noncomputable section

namespace ModernODE

open Filter
open scoped Topology

variable {V E : Type*}
  [SeminormedAddCommGroup V] [NormedSpace ℝ V]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E] [Nontrivial E]

/-- Additional analytic assumptions in Theorem 2. The residual map is an
actual map into Mathlib's interval L² Hilbert space, and its first variation
is the directional derivative of its squared norm. The existence of the
backward-ODE solution and the converging nonlinear family is assumed here and
proved for the IVP in `Theorem2ODE`. The adjoint integral identity is proved
from those ODE data. -/
structure UnstableAnalyticData (T lam : ℝ) (hT : 0 ≤ T)
    (r : V → IntervalL2 E T) (Φ : ℝ → ℝ → E →L[ℝ] E) where
  merit_differentiable : ∀ s v,
    DifferentiableAt ℝ (fun ε : ℝ => residualLoss (r (s + ε • v))) 0
  continuous : ∀ v : E, ContinuousOn (fun t => Φ t 0 v) (Set.Icc 0 T)
  linearizedField : ℝ → E →L[ℝ] E
  homogeneous : ∀ v t, t ∈ Set.Ioo 0 T →
    HasDerivAt (fun u => Φ u 0 v) (linearizedField t (Φ t 0 v)) t
  zero_residual : ∃ x, r x = 0
  adjoint_family : ∀ v : E, ‖v‖ = 1 →
    ∃ (p : ℝ → E) (hp : ContinuousOn p (Set.Icc 0 T)),
      (∀ t ∈ Set.Ioo 0 T, HasDerivAt p
        (-(linearizedField t).adjoint (p t) - rampTrajectory (1 / lam) (fun u => Φ u 0 v) t) t) ∧
      p T = 0 ∧
      ∃ (s : ℕ → V) (ε : ℕ → ℝ) (g : ℕ → IntervalL2 E T),
        Tendsto ε atTop (𝓝 0) ∧
        (∀ᶠ i in atTop, ε i ≠ 0) ∧
        (∀ᶠ i in atTop, r (s i) = ε i • (continuousOn_memLp_two hT hp).toLp p) ∧
        (∀ᶠ i in atTop, BoundsVariation (meritVariation r (s i)) ‖ε i • g i‖) ∧
        Tendsto g atTop (𝓝 (rampTrajectoryL2 (1 / lam) T (fun t => Φ t 0 v) hT (continuous v)))

/-- **Conditional formal version of manuscript Theorem 2.**

The extra analytic interface is fully explicit in `UnstableAnalyticData`.
The proof derives the growing direction, integral energy estimates,
adjoint ratio, limit passage, and exact displayed constant. -/
theorem theorem2 (T lam a b : ℝ) (hT : 0 < T)
    (r : V → IntervalL2 E T) (Φ : ℝ → ℝ → E →L[ℝ] E)
    (ha : 1 ≤ a) (hb : 0 < b) (hlam : 0 < lam) (horizon : 2 ≤ lam * T)
    (A : UnstableAnalyticData T lam hT.le r Φ)
    (hcomp : ∀ t ∈ Set.Icc 0 T, (Φ T t).comp (Φ t 0) = Φ T 0)
    (hupper : ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
      ‖Φ t τ‖ ≤ a * Real.exp (lam * (t - τ)))
    (hlower : b * Real.exp (lam * T) ≤ ‖Φ T 0‖) :
    sInf (Set.range (fun s => residualLoss (r s))) = 0 ∧
    Real.sqrt (optimalPL r (meritVariation r)) ≤
      Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)) := by
  obtain ⟨x, hx⟩ := A.zero_residual
  refine ⟨residualLoss_inf_eq_zero r x hx, ?_⟩
  have ha0 : 0 < a := by linarith
  obtain ⟨v, hv, _, hgrowth⟩ := exists_growing_direction Φ a b lam T ha0 hcomp hupper hlower
  let z : ℝ → E := fun t => Φ t 0 v
  let η := rampTrajectoryL2 (1 / lam) T z hT.le (A.continuous v)
  let u := rampForcingL2 (1 / lam) T z hT.le (A.continuous v)
  have hδ : 0 < 1 / lam := by positivity
  have hδT : 1 / lam < T := by
    apply (div_lt_iff₀ hlam).2
    nlinarith
  have hcancel : 2 * lam * (1 / lam) = 2 := by field_simp
  have hcoeff : a ^ 2 / (2 * lam * (1 / lam) ^ 2) = a ^ 2 * lam / 2 := by
    field_simp
  have henergy := ramp_energy_bounds z a b lam (1 / lam) T
    ha0 hb hlam hδ hδT (A.continuous v) hgrowth
  have hU : ‖u‖ ^ 2 ≤ (a ^ 2 * lam / 2) * (Real.exp 2 - 1) := by
    rw [norm_rampForcingL2_sq]
    simpa only [hcancel, hcoeff] using henergy.1
  have hE : b ^ 2 / (2 * lam * a ^ 2) *
      (Real.exp (2 * lam * T) - Real.exp 2) ≤ ‖η‖ ^ 2 := by
    rw [norm_rampTrajectoryL2_sq]
    simpa only [hcancel] using henergy.2
  obtain ⟨p, hp, hpODE, hpT, s, ε, g, _, hε, hr, hG, hlim⟩ := A.adjoint_family v hv
  let pL2 := (continuousOn_memLp_two hT.le hp).toLp p
  have hdual : ‖η‖ ^ 2 = inner ℝ pL2 u :=
    ramp_adjoint_duality (1 / lam) T z p A.linearizedField hδ hδT
      (A.continuous v) hp (A.homogeneous v) hpODE hpT
  exact theorem2_of_energy_and_scaled_family r (meritVariation r) a b lam T
    ha hb hlam horizon η pL2 u hU hE hdual s ε g hε hr hG hlim

end ModernODE
