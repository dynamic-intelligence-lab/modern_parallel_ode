import ModernODE.PLTransfer
import ModernODE.Unstable

/-!
# Unstable PL bound with an explicit analytic interface

The hypotheses expose the energy estimates, adjoint duality, and local
nonlinear family used by the manuscript. They do not assume a PL upper bound.
-/

namespace ModernODE

open Filter
open scoped Topology

variable {S V H ι : Type*} [SeminormedAddCommGroup V]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- A conditional version of Theorem 2. The analytic hypotheses are energy
bounds for a ramp and its forcing, adjoint duality, and an exact-residual
family with converging scaled first-variation bounds. -/
theorem theorem2_of_energy_and_scaled_family
    (r : S → H) (d : S → V → ℝ) (a b lam T : ℝ)
    (ha : 1 ≤ a) (hb : 0 < b) (hlam : 0 < lam) (horizon : 2 ≤ lam * T)
    (η p u : H)
    (hU : ‖u‖ ^ 2 ≤ (a ^ 2 * lam / 2) * (Real.exp 2 - 1))
    (hE : b ^ 2 / (2 * lam * a ^ 2) *
      (Real.exp (2 * lam * T) - Real.exp 2) ≤ ‖η‖ ^ 2)
    (hdual : ‖η‖ ^ 2 = inner ℝ p u)
    {l : Filter ι} [l.NeBot] (s : ι → S) (ε : ι → ℝ) (g : ι → H)
    (hε : ∀ᶠ i in l, ε i ≠ 0)
    (hr : ∀ᶠ i in l, r (s i) = ε i • p)
    (hG : ∀ᶠ i in l, BoundsVariation (d (s i)) ‖ε i • g i‖)
    (hlim : Tendsto g l (𝓝 η)) :
    Real.sqrt (optimalPL r d) ≤
      Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)) := by
  have ha0 : 0 < a := by linarith
  have hdiff : 0 < Real.exp (2 * lam * T) - Real.exp 2 := by
    apply sub_pos.mpr
    apply Real.exp_lt_exp.mpr
    linarith
  have hηsq : 0 < ‖η‖ ^ 2 := lt_of_lt_of_le (by positivity) hE
  have hη : η ≠ 0 := by
    intro hz
    simp [hz] at hηsq
  have hp : ‖p‖ ≠ 0 := by
    intro hz
    have hpz : p = 0 := norm_eq_zero.mp hz
    simp only [hpz, inner_zero_left] at hdual
    linarith
  have hlocal := optimalPL_le_of_scaled_family s ε g p η hp hε hr hG hlim
  have hadj := adjoint_ratio_from_duality η p u hη hdual
  have henergy := unstable_energy_ratio_at_inverse_rate a b lam T _ _
    ha0 hb hlam horizon hU hE
  exact unstable_sqrt_constant a b lam T _ hb hlam horizon (hlocal.trans (hadj.trans henergy))

end ModernODE
