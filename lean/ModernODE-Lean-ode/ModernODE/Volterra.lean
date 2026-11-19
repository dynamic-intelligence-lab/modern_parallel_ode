import ModernODE.Stable
import ModernODE.Schur

/-!
# Exponential Volterra kernel

This file proves both mass bounds for the causal exponential kernel, and
therefore its L² energy gain via the Schur estimate. Trajectories are treated
as functions on the real line, with the kernel zero outside `0 ≤ τ ≤ t ≤ T`.
-/

noncomputable section

namespace ModernODE

open MeasureTheory Set

/-- The stable exponential majorant, extended by zero outside its time triangle. -/
def volterraKernel (a c T t τ : ℝ) : ℝ :=
  if 0 ≤ τ ∧ τ ≤ t ∧ t ≤ T then a * Real.exp (-c * (t - τ)) else 0

lemma volterraKernel_nonneg {a c T t τ : ℝ} (ha : 0 ≤ a) :
    0 ≤ volterraKernel a c T t τ := by
  unfold volterraKernel
  split_ifs <;> positivity

lemma volterraKernel_row {a c T t : ℝ} (ht : t ∈ Icc 0 T) :
    volterraKernel a c T t =
      (Icc 0 t).indicator (fun τ => a * Real.exp (-c * (t - τ))) := by
  funext τ
  simp [volterraKernel, Set.indicator_apply, ht.2]

lemma volterraKernel_row_zero {a c T t : ℝ} (ht : t ∉ Icc 0 T) :
    volterraKernel a c T t = 0 := by
  funext τ
  have hfalse : ¬ (0 ≤ τ ∧ τ ≤ t ∧ t ≤ T) := by
    intro h
    exact ht ⟨h.1.trans h.2.1, h.2.2⟩
  simp [volterraKernel, hfalse]

lemma volterraKernel_col {a c T τ : ℝ} (hτ : τ ∈ Icc 0 T) :
    (fun t => volterraKernel a c T t τ) =
      (Icc τ T).indicator (fun t => a * Real.exp (-c * (t - τ))) := by
  funext t
  simp [volterraKernel, Set.indicator_apply, hτ.1]

lemma volterraKernel_col_zero {a c T τ : ℝ} (hτ : τ ∉ Icc 0 T) :
    (fun t => volterraKernel a c T t τ) = 0 := by
  funext t
  have hfalse : ¬ (0 ≤ τ ∧ τ ≤ t ∧ t ≤ T) := by
    intro h
    exact hτ ⟨h.1, h.2.1.trans h.2.2⟩
  simp [volterraKernel, hfalse]

/-- Increasing the horizon increases the kernel mass. -/
theorem stableGain_mono {a c S T : ℝ} (ha : 0 ≤ a) (hc : 0 < c) (hST : S ≤ T) :
    stableGain a c S ≤ stableGain a c T := by
  unfold stableGain
  apply div_le_div_of_nonneg_right _ hc.le
  apply mul_le_mul_of_nonneg_left _ ha
  have hexp : Real.exp (-c * T) ≤ Real.exp (-c * S) := by
    apply Real.exp_le_exp.mpr
    nlinarith
  linarith

/-- Each kernel row is integrable, because it is a continuous function
restricted to a compact interval. -/
theorem volterraKernel_integrable (a c T t : ℝ) :
    Integrable (volterraKernel a c T t) := by
  by_cases ht : t ∈ Icc 0 T
  · rw [volterraKernel_row ht]
    apply IntegrableOn.integrable_indicator _ measurableSet_Icc
    apply Continuous.integrableOn_Icc
    fun_prop
  · rw [volterraKernel_row_zero ht]
    exact integrable_zero ℝ ℝ volume

/-- Exact row mass on the time interval. -/
theorem volterraKernel_row_integral {a c T t : ℝ} (hc : c ≠ 0) (ht : t ∈ Icc 0 T) :
    (∫ τ, volterraKernel a c T t τ) = stableGain a c t := by
  rw [volterraKernel_row ht, integral_indicator measurableSet_Icc,
    integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le ht.1]
  change (∫ τ in (0 : ℝ)..t, (fun u => a * Real.exp (-c * u)) (t - τ)) = _
  rw [intervalIntegral.integral_comp_sub_left (fun u => a * Real.exp (-c * u)) t]
  simp only [sub_self, sub_zero]
  exact stableGain_integral hc

/-- Exact column mass on the time interval. -/
theorem volterraKernel_col_integral {a c T τ : ℝ} (hc : c ≠ 0) (hτ : τ ∈ Icc 0 T) :
    (∫ t, volterraKernel a c T t τ) = stableGain a c (T - τ) := by
  rw [volterraKernel_col hτ, integral_indicator measurableSet_Icc,
    integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le hτ.2]
  change (∫ t in τ..T, (fun u => a * Real.exp (-c * u)) (t - τ)) = _
  rw [intervalIntegral.integral_comp_sub_right (fun u => a * Real.exp (-c * u)) τ]
  simp only [sub_self]
  exact stableGain_integral hc

/-- Every row of the exponential Volterra kernel has mass at most `B_T`. -/
theorem volterraKernel_row_bound {a c T : ℝ}
    (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T) (t : ℝ) :
    (∫ τ, volterraKernel a c T t τ) ≤ stableGain a c T := by
  by_cases ht : t ∈ Icc 0 T
  · rw [volterraKernel_row_integral (ne_of_gt hc) ht]
    exact stableGain_mono (by linarith) hc ht.2
  · rw [volterraKernel_row_zero ht]
    simp only [Pi.zero_apply, integral_zero]
    exact (stableGain_pos ha hc hT).le

/-- Every column of the exponential Volterra kernel has mass at most `B_T`. -/
theorem volterraKernel_col_bound {a c T : ℝ}
    (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T) (τ : ℝ) :
    (∫ t, volterraKernel a c T t τ) ≤ stableGain a c T := by
  by_cases hτ : τ ∈ Icc 0 T
  · rw [volterraKernel_col_integral (ne_of_gt hc) hτ]
    exact stableGain_mono (by linarith) hc (by linarith [hτ.1])
  · rw [volterraKernel_col_zero hτ, integral_zero']
    exact (stableGain_pos ha hc hT).le

/-- Exponential kernel domination implies the finite-horizon L² gain.
Only analytic integrability and pointwise domination are assumed. -/
theorem volterra_energy_bound (r η : ℝ → ℝ) {a c T : ℝ}
    (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T)
    (hr_nonneg : ∀ t, 0 ≤ r t) (hη_nonneg : ∀ t, 0 ≤ η t)
    (hdom : ∀ t, η t ≤ ∫ τ, volterraKernel a c T t τ * r τ)
    (hkr : ∀ t, Integrable (fun τ => volterraKernel a c T t τ * r τ))
    (hkr2 : ∀ t, Integrable (fun τ => volterraKernel a c T t τ * (r τ) ^ 2))
    (hprod : Integrable (fun p : ℝ × ℝ => volterraKernel a c T p.1 p.2 * (r p.2) ^ 2)
      (volume.prod volume))
    (hη2 : Integrable (fun t => (η t) ^ 2))
    (hr2 : Integrable (fun t => (r t) ^ 2)) :
    (∫ t, (η t) ^ 2) ≤ stableGain a c T ^ 2 * ∫ t, (r t) ^ 2 := by
  exact schur_energy_bound (volterraKernel a c T) r η (stableGain_pos ha hc hT)
    (fun _ _ => volterraKernel_nonneg (by linarith)) hr_nonneg hη_nonneg
    (volterraKernel_row_bound ha hc hT) (volterraKernel_col_bound ha hc hT)
    hdom (volterraKernel_integrable a c T) hkr hkr2 hprod hη2 hr2

end ModernODE
