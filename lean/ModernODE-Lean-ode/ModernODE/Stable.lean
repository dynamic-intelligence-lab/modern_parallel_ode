import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Tactic

/-!
# Stable PL estimates from a bounded residual correction

This file formalizes the duality and constant calculations in the stable PL
argument. The construction of the correction from the ODE, its integral gain
estimate, and the identification of the first variation with the L² pairing
are hypotheses of these results, not conclusions proved here.
-/

noncomputable section

namespace ModernODE

open scoped InnerProductSpace

/-- The elementary duality argument, stated so that it applies to a bound on
all directional derivatives, whether or not a Hilbert-space gradient has
already been constructed. -/
theorem residual_le_gain_mul_slope
    {R E G B : ℝ} (hR : 0 ≤ R) (hG : 0 ≤ G) (hB : 0 ≤ B)
    (hdual : R ^ 2 ≤ G * E) (hgain : E ≤ B * R) :
    R ≤ B * G := by
  by_cases hzero : R = 0
  · subst R
    exact mul_nonneg hB hG
  · have hRpos : 0 < R := lt_of_le_of_ne hR (Ne.symm hzero)
    apply (mul_le_mul_iff_left₀ hRpos).mp
    have h := hdual.trans (mul_le_mul_of_nonneg_left hgain hG)
    nlinarith only [h]

/-- A correction of norm at most `B * ‖r‖`, paired against the gradient,
forces the gradient norm to be at least `‖r‖ / B`. -/
theorem residual_le_gain_mul_gradient
    {H R : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    [NormedAddCommGroup R]
    (r : R) (g η : H) {B : ℝ} (hB : 0 ≤ B)
    (hpair : ⟪g, η⟫_ℝ = ‖r‖ ^ 2)
    (hgain : ‖η‖ ≤ B * ‖r‖) :
    ‖r‖ ≤ B * ‖g‖ := by
  apply residual_le_gain_mul_slope (norm_nonneg r) (norm_nonneg g) hB
    (E := ‖η‖)
  · rw [← hpair]
    exact real_inner_le_norm g η
  · exact hgain

/-- Squared-residual PL inequality from a correction and its gain bound. -/
theorem correction_implies_pl
    {H R : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    [NormedAddCommGroup R]
    (r : R) (g η : H) {B : ℝ} (hB : 0 < B)
    (hpair : ⟪g, η⟫_ℝ = ‖r‖ ^ 2)
    (hgain : ‖η‖ ≤ B * ‖r‖) :
    (1 / B) ^ 2 * (‖r‖ ^ 2 / 2) ≤ ‖g‖ ^ 2 / 2 := by
  have h := residual_le_gain_mul_gradient r g η hB.le hpair hgain
  have hdiv : ‖r‖ / B ≤ ‖g‖ := by
    apply (div_le_iff₀ hB).2
    simpa [mul_comm] using h
  have hsquare : (‖r‖ / B) ^ 2 ≤ ‖g‖ ^ 2 :=
    (sq_le_sq₀ (div_nonneg (norm_nonneg r) hB.le) (norm_nonneg g)).2 hdiv
  calc
    (1 / B) ^ 2 * (‖r‖ ^ 2 / 2) = (‖r‖ / B) ^ 2 / 2 := by ring
    _ ≤ ‖g‖ ^ 2 / 2 := by linarith

/-- The finite-horizon gain appearing in Theorem 1. -/
def stableGain (a c T : ℝ) : ℝ := a * (1 - Real.exp (-c * T)) / c

/-- Direct evaluation of the exponential kernel's total mass. -/
theorem stableGain_integral {a c T : ℝ} (hc : c ≠ 0) :
    (∫ u in (0 : ℝ)..T, a * Real.exp (-c * u)) = stableGain a c T := by
  rw [intervalIntegral.integral_const_mul,
    intervalIntegral.integral_comp_mul_left Real.exp (neg_ne_zero.mpr hc),
    integral_exp]
  simp only [mul_zero, Real.exp_zero, smul_eq_mul]
  unfold stableGain
  simp only [inv_neg, div_eq_mul_inv]
  ring

theorem stableGain_pos {a c T : ℝ} (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T) :
    0 < stableGain a c T := by
  have hexp : Real.exp (-c * T) < 1 := by
    apply Real.exp_lt_one_iff.mpr
    nlinarith
  unfold stableGain
  exact div_pos (mul_pos (by linarith) (by linarith)) hc

/-- The reciprocal gain has exactly the constant used in the manuscript. -/
theorem stableGain_inv {a c T : ℝ} (ha : 1 ≤ a) (hc : 0 < c) (_hT : 0 < T) :
    1 / stableGain a c T = (c / a) / (1 - Real.exp (-c * T)) := by
  unfold stableGain
  field_simp

/-- The finite-horizon stable constant dominates the horizon-independent one. -/
theorem stableGain_inv_ge {a c T : ℝ} (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T) :
    c / a ≤ 1 / stableGain a c T := by
  have ha0 : a ≠ 0 := by linarith
  have hc0 : c ≠ 0 := ne_of_gt hc
  apply (le_div_iff₀ (stableGain_pos ha hc hT)).2
  calc
    c / a * stableGain a c T = 1 - Real.exp (-c * T) := by
      unfold stableGain
      field_simp
    _ ≤ 1 := by linarith [Real.exp_pos (-c * T)]

/-- Conditional stable PL inequality: the analytic correction and duality
identities are explicitly supplied as hypotheses. -/
theorem stable_correction_implies_pl
    {H R : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]
    [NormedAddCommGroup R]
    (r : R) (g η : H) {a c T : ℝ}
    (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T)
    (hpair : ⟪g, η⟫_ℝ = ‖r‖ ^ 2)
    (hgain : ‖η‖ ≤ stableGain a c T * ‖r‖) :
    ((c / a) / (1 - Real.exp (-c * T))) ^ 2 * (‖r‖ ^ 2 / 2) ≤
      ‖g‖ ^ 2 / 2 := by
  rw [← stableGain_inv ha hc hT]
  exact correction_implies_pl r g η (stableGain_pos ha hc hT) hpair hgain

/-- Conversion from a squared lower bound on any `μ` to both square-root
bounds displayed in Theorem 1. This is only the scalar conversion. -/
theorem stable_sqrt_bound {μ a c T : ℝ}
    (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T)
    (hμ : (1 / stableGain a c T) ^ 2 ≤ μ) :
    (c / a) / (1 - Real.exp (-c * T)) ≤ Real.sqrt μ ∧
      c / a ≤ (c / a) / (1 - Real.exp (-c * T)) ∧ 0 < c / a := by
  have hμ0 : 0 ≤ μ := le_trans (sq_nonneg _) hμ
  have hi : 0 < 1 / stableGain a c T := one_div_pos.mpr (stableGain_pos ha hc hT)
  have hbound : 1 / stableGain a c T ≤ Real.sqrt μ := by
    apply (sq_le_sq₀ hi.le (Real.sqrt_nonneg μ)).1
    simpa only [Real.sq_sqrt hμ0] using hμ
  rw [stableGain_inv ha hc hT] at hbound
  refine ⟨hbound, ?_, div_pos hc (by linarith)⟩
  rw [← stableGain_inv ha hc hT]
  exact stableGain_inv_ge ha hc hT

end ModernODE
