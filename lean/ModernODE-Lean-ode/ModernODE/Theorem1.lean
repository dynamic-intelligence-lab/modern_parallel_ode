import ModernODE.StablePL
import ModernODE.Merit
import ModernODE.Transition

/-!
# Theorem 1 with an explicit analytic interface

This conditional theorem proves the complete displayed stable bound from
the exponential transition estimate. It does not assume the L² gain or the
PL conclusion. `StableAnalyticData` lists the additional analytic hypotheses
connecting abstract trajectory/residual spaces to actual time functions.
-/

noncomputable section

namespace ModernODE

open MeasureTheory

variable {V H E : Type*}
  [SeminormedAddCommGroup V] [NormedSpace ℝ V]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]
  [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Additional analytic assumptions for Theorem 1. `V` carries the trajectory
L² norm, and may be an incomplete space of zero-initial-value variations.
The residual path and correction paths are extended by zero outside `[0,T]`.
For the IVP this data is constructed in `ODEInstance` (`theorem1_ode`). -/
structure StableAnalyticData (r : V → H) (a c T : ℝ) where
  merit_differentiable : ∀ s v,
    DifferentiableAt ℝ (fun ε : ℝ => residualLoss (r (s + ε • v))) 0
  residualPath : V → ℝ → E
  residual_support : ∀ s t, t ∉ Set.Icc 0 T → residualPath s t = 0
  transition : V → ℝ → ℝ → E →L[ℝ] E
  residual_energy : ∀ s, ‖r s‖ ^ 2 = ∫ t, ‖residualPath s t‖ ^ 2
  residual_sq_integrable : ∀ s, Integrable (fun t => ‖residualPath s t‖ ^ 2)
  kernel_residual_integrable : ∀ s t,
    Integrable (fun τ => volterraKernel a c T t τ * ‖residualPath s τ‖)
  kernel_residual_sq_integrable : ∀ s t,
    Integrable (fun τ => volterraKernel a c T t τ * ‖residualPath s τ‖ ^ 2)
  joint_integrable : ∀ s,
    Integrable (fun p : ℝ × ℝ => volterraKernel a c T p.1 p.2 * ‖residualPath s p.2‖ ^ 2)
      (volume.prod volume)
  correction : ∀ s, ∃ (η : V) (ηpath : ℝ → E),
    HasDerivAt (fun ε : ℝ => r (s + ε • η)) (r s) 0 ∧
    ‖η‖ ^ 2 = (∫ t, ‖ηpath t‖ ^ 2) ∧
    (∀ t, ηpath t = ∫ τ, transitionAction (transition s) T (residualPath s) t τ) ∧
    Integrable (fun t => ‖ηpath t‖ ^ 2)
  zero_residual : ∃ x, r x = 0
  finite_nonzero_point : ∃ s G,
    BoundsVariation (meritVariation r s) G ∧ 0 < ‖r s‖ ^ 2

/-- **Conditional formal version of manuscript Theorem 1.**

Besides the manuscript's exponential transition bound, this theorem requires
`StableAnalyticData`: time-function representations, integrability, a
variation-of-constants correction whose residual derivative is the residual,
a zero-residual solution, and one finite-slope nonzero-residual point.
It proves zero minimum, the global extended-norm PL inequality, and both
square-root bounds. -/
theorem theorem1 (r : V → H) (a c T : ℝ)
    (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T)
    (A : StableAnalyticData (E := E) r a c T)
    (hdecay : ∀ s τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
      ‖A.transition s t τ‖ ≤ a * Real.exp (-c * (t - τ))) :
    sInf (Set.range (fun s => residualLoss (r s))) = 0 ∧
    IsPL r (meritVariation r) (((c / a) / (1 - Real.exp (-c * T))) ^ 2) ∧
    (c / a) / (1 - Real.exp (-c * T)) ≤ Real.sqrt (optimalPL r (meritVariation r)) ∧
    c / a ≤ (c / a) / (1 - Real.exp (-c * T)) ∧ 0 < c / a := by
  obtain ⟨x, hx⟩ := A.zero_residual
  refine ⟨residualLoss_inf_eq_zero r x hx, ?_⟩
  apply theorem1_of_correction_estimate r (meritVariation r) ha hc hT
  · intro s
    obtain ⟨η, ηpath, hderiv, henergy, hformula, hηint⟩ := A.correction s
    refine ⟨η, meritVariation_correction r s η hderiv, ?_⟩
    have hdom := correction_dominated_by_volterra (A.transition s) (A.residualPath s)
      ηpath a c T (hdecay s) hformula (A.kernel_residual_integrable s)
    have hgain := volterra_energy_bound
      (fun t => ‖A.residualPath s t‖) (fun t => ‖ηpath t‖)
      ha hc hT (fun _ => norm_nonneg _) (fun _ => norm_nonneg _) hdom
      (A.kernel_residual_integrable s) (A.kernel_residual_sq_integrable s)
      (A.joint_integrable s) hηint (A.residual_sq_integrable s)
    rw [← henergy, ← A.residual_energy s] at hgain
    have hB := (stableGain_pos ha hc hT).le
    have hnorm : 0 ≤ stableGain a c T * ‖r s‖ := mul_nonneg hB (norm_nonneg _)
    nlinarith [norm_nonneg η]
  · obtain ⟨s, G, hG, hr⟩ := A.finite_nonzero_point
    exact plConstants_bddAbove s G hG hr

end ModernODE
