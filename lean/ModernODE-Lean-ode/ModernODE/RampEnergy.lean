import Mathlib

/-!
# Integral energy estimates for the unstable ramp perturbation

The trajectory and forcing below are actual functions, and their energies
are Lebesgue interval integrals. The estimates are derived from pointwise
growth bounds, rather than assumed as scalar hypotheses. Identifying the
forcing with the weak ODE residual remains a separate analytic obligation.
-/

noncomputable section

open MeasureTheory Set

namespace ModernODE

def rampTrajectory {E : Type*} [SMul ℝ E] (δ : ℝ) (z : ℝ → E) (t : ℝ) : E :=
  min (t / δ) 1 • z t

def rampForcing {E : Type*} [Zero E] [SMul ℝ E] (δ : ℝ) (z : ℝ → E) (t : ℝ) : E :=
  if t ≤ δ then δ⁻¹ • z t else 0

def trajectoryEnergy {E : Type*} [Norm E] (f : ℝ → E) (T : ℝ) : ℝ :=
  ∫ t in 0..T, ‖f t‖ ^ 2

theorem integral_exp_linear (k p q : ℝ) (hk : k ≠ 0) :
    (∫ t in p..q, Real.exp (k * t)) =
      (Real.exp (k * q) - Real.exp (k * p)) / k := by
  rw [intervalIntegral.integral_comp_mul_left Real.exp hk, integral_exp]
  simp only [smul_eq_mul, div_eq_mul_inv, mul_comm]

theorem rampTrajectory_eq_of_le
    {E : Type*} [MulAction ℝ E] (δ : ℝ) (z : ℝ → E) (hδ : 0 < δ)
    {t : ℝ} (ht : δ ≤ t) : rampTrajectory δ z t = z t := by
  have hratio : 1 ≤ t / δ := (le_div_iff₀ hδ).2 (by simpa using ht)
  simp [rampTrajectory, min_eq_right hratio]

/-- The forcing acts only on the initial interval. Endpoint values do not
affect this identity. -/
theorem ramp_forcing_energy_eq
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ T : ℝ) (z : ℝ → E) (hδ : 0 < δ) (hδT : δ ≤ T) :
    trajectoryEnergy (rampForcing δ z) T =
      δ⁻¹ ^ 2 * ∫ t in 0..δ, ‖z t‖ ^ 2 := by
  unfold trajectoryEnergy
  calc
    (∫ t in 0..T, ‖rampForcing δ z t‖ ^ 2) =
        ∫ t in 0..T, ({t : ℝ | t ≤ δ}.indicator
          (fun t => ‖δ⁻¹ • z t‖ ^ 2)) t := by
      apply intervalIntegral.integral_congr
      intro t ht
      by_cases htd : t ≤ δ <;> simp [rampForcing, htd]
    _ = ∫ t in 0..δ, ‖δ⁻¹ • z t‖ ^ 2 :=
      intervalIntegral.integral_indicator ⟨hδ.le, hδT⟩
    _ = δ⁻¹ ^ 2 * ∫ t in 0..δ, ‖z t‖ ^ 2 := by
      simp_rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hδ), mul_pow]
      exact intervalIntegral.integral_const_mul _ _

/-- Actual integral estimates in Step 2 of Theorem 2. No integral-energy
estimate is assumed: only continuity and pointwise growth bounds on z. -/
theorem ramp_energy_bounds
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (z : ℝ → E) (a b lam δ T : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hlam : 0 < lam) (hδ : 0 < δ) (hδT : δ < T)
    (hz : ContinuousOn z (Icc 0 T))
    (hbound : ∀ t ∈ Icc 0 T,
      b / a * Real.exp (lam * t) ≤ ‖z t‖ ∧
      ‖z t‖ ≤ a * Real.exp (lam * t)) :
    trajectoryEnergy (rampForcing δ z) T ≤
      a ^ 2 / (2 * lam * δ ^ 2) * (Real.exp (2 * lam * δ) - 1) ∧
    b ^ 2 / (2 * lam * a ^ 2) *
        (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ)) ≤
      trajectoryEnergy (rampTrajectory δ z) T := by
  have hT : 0 < T := hδ.trans hδT
  have h2lam : 2 * lam ≠ 0 := by positivity
  have hexp : ∀ t : ℝ, Real.exp (lam * t) ^ 2 = Real.exp (2 * lam * t) := by
    intro t
    rw [pow_two, ← Real.exp_add]
    congr 1
    ring
  have hz0δ : ContinuousOn (fun t => ‖z t‖ ^ 2) (Icc 0 δ) :=
    (hz.mono (Icc_subset_Icc le_rfl hδT.le)).norm.pow 2
  have hη : ContinuousOn (rampTrajectory δ z) (Icc 0 T) := by
    exact ((continuous_id.div_const δ).min continuous_const).continuousOn.smul hz
  have hηenergy : IntervalIntegrable (fun t => ‖rampTrajectory δ z t‖ ^ 2)
      volume 0 T :=
    (hη.norm.pow 2).intervalIntegrable_of_Icc hT.le
  have hηδT : IntervalIntegrable (fun t => ‖rampTrajectory δ z t‖ ^ 2)
      volume δ T :=
    ((hη.mono (Icc_subset_Icc hδ.le le_rfl)).norm.pow 2).intervalIntegrable_of_Icc hδT.le
  constructor
  · rw [ramp_forcing_energy_eq δ T z hδ hδT.le]
    have hintegral : (∫ t in 0..δ, ‖z t‖ ^ 2) ≤
        ∫ t in 0..δ, a ^ 2 * Real.exp (2 * lam * t) := by
      apply intervalIntegral.integral_mono_on hδ.le
        (hz0δ.intervalIntegrable_of_Icc hδ.le)
        ((by fun_prop : Continuous (fun t : ℝ => a ^ 2 * Real.exp (2 * lam * t))).intervalIntegrable 0 δ)
      intro t ht
      have hp := (hbound t ⟨ht.1, ht.2.trans hδT.le⟩).2
      have hsq : ‖z t‖ ^ 2 ≤ (a * Real.exp (lam * t)) ^ 2 :=
        (sq_le_sq₀ (norm_nonneg (z t)) (by positivity)).2 hp
      simpa only [mul_pow, hexp] using hsq
    calc
      δ⁻¹ ^ 2 * (∫ t in 0..δ, ‖z t‖ ^ 2) ≤
          δ⁻¹ ^ 2 * (∫ t in 0..δ, a ^ 2 * Real.exp (2 * lam * t)) :=
        mul_le_mul_of_nonneg_left hintegral (sq_nonneg _)
      _ = a ^ 2 / (2 * lam * δ ^ 2) * (Real.exp (2 * lam * δ) - 1) := by
        rw [intervalIntegral.integral_const_mul, integral_exp_linear (2 * lam) 0 δ h2lam]
        simp only [mul_zero, Real.exp_zero]
        field_simp
  · have hpoint : (∫ t in δ..T, (b / a) ^ 2 * Real.exp (2 * lam * t)) ≤
        ∫ t in δ..T, ‖rampTrajectory δ z t‖ ^ 2 := by
      apply intervalIntegral.integral_mono_on hδT.le
        ((by fun_prop : Continuous (fun t : ℝ => (b / a) ^ 2 *
          Real.exp (2 * lam * t))).intervalIntegrable δ T) hηδT
      intro t ht
      rw [rampTrajectory_eq_of_le δ z hδ ht.1]
      have hp := (hbound t ⟨hδ.le.trans ht.1, ht.2⟩).1
      have hsq : (b / a * Real.exp (lam * t)) ^ 2 ≤ ‖z t‖ ^ 2 :=
        (sq_le_sq₀ (by positivity) (norm_nonneg (z t))).2 hp
      simpa only [mul_pow, hexp] using hsq
    calc
      b ^ 2 / (2 * lam * a ^ 2) *
          (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ)) =
          ∫ t in δ..T, (b / a) ^ 2 * Real.exp (2 * lam * t) := by
        rw [intervalIntegral.integral_const_mul, integral_exp_linear (2 * lam) δ T h2lam]
        field_simp
      _ ≤ ∫ t in δ..T, ‖rampTrajectory δ z t‖ ^ 2 := hpoint
      _ ≤ trajectoryEnergy (rampTrajectory δ z) T := by
        apply intervalIntegral.integral_mono_interval hδ.le hδT.le le_rfl
          _ hηenergy
        exact Filter.Eventually.of_forall (fun t => sq_nonneg _)

end ModernODE
