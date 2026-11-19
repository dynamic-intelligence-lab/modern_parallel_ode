import ModernODE.Core
import ModernODE.Stable

/-!
# Stable PL theorem with explicit analytic hypotheses

The ODE-to-correction construction is not asserted here. The hypotheses say
exactly which first-variation identity and norm estimate an application must
supply. The extended directional norm definition also covers candidates at
which no L² gradient exists.
-/

namespace ModernODE

variable {S V H : Type*} [SeminormedAddCommGroup V] [SeminormedAddCommGroup H]

/-- Uniform residual corrections imply a global PL inequality, including
trajectories with unbounded first variation. -/
theorem isPL_of_corrections (r : S → H) (d : S → V → ℝ)
    {B : ℝ} (hB : 0 < B)
    (hcorrection : ∀ s, ∃ η : V, d s η = ‖r s‖ ^ 2 ∧ ‖η‖ ≤ B * ‖r s‖) :
    IsPL r d ((1 / B) ^ 2) := by
  refine ⟨sq_nonneg _, ?_⟩
  intro s G hG
  obtain ⟨η, hpair, hgain⟩ := hcorrection s
  have hdual : ‖r s‖ ^ 2 ≤ G * ‖η‖ := by
    calc
      ‖r s‖ ^ 2 = |d s η| := by rw [hpair, abs_of_nonneg (sq_nonneg _)]
      _ ≤ G * ‖η‖ := hG.2 η
  have hnorm := residual_le_gain_mul_slope (norm_nonneg (r s)) hG.1 hB.le hdual hgain
  have hdiv : ‖r s‖ / B ≤ G := by
    apply (div_le_iff₀ hB).2
    simpa [mul_comm] using hnorm
  have hsquare : (‖r s‖ / B) ^ 2 ≤ G ^ 2 := by
    nlinarith [div_nonneg (norm_nonneg (r s)) hB.le]
  calc
    (1 / B) ^ 2 * ‖r s‖ ^ 2 = (‖r s‖ / B) ^ 2 := by ring
    _ ≤ G ^ 2 := hsquare

/-- Conditional version of the displayed Theorem 1 bounds. This theorem
assumes the analytic correction identity and gain estimate. It also explicitly
assumes boundedness of the set of constants to use a real-valued supremum. -/
theorem theorem1_of_correction_estimate (r : S → H) (d : S → V → ℝ)
    {a c T : ℝ} (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T)
    (hcorrection : ∀ s, ∃ η : V,
      d s η = ‖r s‖ ^ 2 ∧ ‖η‖ ≤ stableGain a c T * ‖r s‖)
    (hbounded : BddAbove {μ | IsPL r d μ}) :
    IsPL r d (((c / a) / (1 - Real.exp (-c * T))) ^ 2) ∧
    (c / a) / (1 - Real.exp (-c * T)) ≤ Real.sqrt (optimalPL r d) ∧
    c / a ≤ (c / a) / (1 - Real.exp (-c * T)) ∧ 0 < c / a := by
  have hPL := isPL_of_corrections r d (stableGain_pos ha hc hT) hcorrection
  have hopt := le_optimalPL hbounded hPL
  have hsqrt := stable_sqrt_bound ha hc hT hopt
  rw [stableGain_inv ha hc hT] at hPL
  exact ⟨hPL, hsqrt⟩

end ModernODE
