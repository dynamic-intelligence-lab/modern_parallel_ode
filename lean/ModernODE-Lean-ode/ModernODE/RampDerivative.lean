import ModernODE.RampEnergy

/-!
# The ramp's linearized ODE residual

At every time except the ramp corner, differentiating the actual ramp
trajectory gives the forcing used in the energy estimates.
-/

open Filter
open scoped Topology

namespace ModernODE

theorem hasDerivAt_rampTrajectory_of_ne
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ : ℝ) (hδ : 0 < δ) (z : ℝ → E) (A : ℝ → E →L[ℝ] E)
    {t : ℝ} (ht : t ≠ δ) (hz : HasDerivAt z (A t (z t)) t) :
    HasDerivAt (rampTrajectory δ z)
      (A t (rampTrajectory δ z t) + rampForcing δ z t) t := by
  rcases lt_or_gt_of_ne ht with hlt | hgt
  · have hratio : t / δ ≤ 1 := (div_le_iff₀ hδ).2 (by simpa using hlt.le)
    have heq : rampTrajectory δ z =ᶠ[𝓝 t] (fun s => (s / δ) • z s) := by
      filter_upwards [eventually_lt_nhds hlt] with s hs
      have hsratio : s / δ ≤ 1 := (div_le_iff₀ hδ).2 (by simpa using hs.le)
      simp [rampTrajectory, min_eq_left hsratio]
    have hd := (((hasDerivAt_id t).div_const δ).fun_smul hz).congr_of_eventuallyEq heq
    simpa [rampTrajectory, min_eq_left hratio, rampForcing, hlt.le, map_smul,
      one_div] using hd
  · have heq : rampTrajectory δ z =ᶠ[𝓝 t] z := by
      filter_upwards [eventually_gt_nhds hgt] with s hs
      exact rampTrajectory_eq_of_le δ z hδ hs.le
    have hd := hz.congr_of_eventuallyEq heq
    simpa [rampTrajectory_eq_of_le δ z hδ hgt.le, rampForcing, not_le.mpr hgt]
      using hd

/-- The ramp perturbation satisfies its inhomogeneous linearized equation
almost everywhere on the open time interval. The corner is a null set. -/
theorem ae_hasDerivAt_rampTrajectory
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ : ℝ) (hδ : 0 < δ) (z : ℝ → E) (A : ℝ → E →L[ℝ] E) (T : ℝ)
    (hz : ∀ t ∈ Set.Ioo 0 T, HasDerivAt z (A t (z t)) t) :
    ∀ᵐ t ∂MeasureTheory.volume.restrict (Set.Ioo 0 T),
      HasDerivAt (rampTrajectory δ z)
        (A t (rampTrajectory δ z t) + rampForcing δ z t) t := by
  have hne0 : ∀ᵐ t : ℝ ∂MeasureTheory.volume, t ≠ δ := by
    rw [MeasureTheory.ae_iff]
    simp
  have hne : ∀ᵐ t ∂MeasureTheory.volume.restrict (Set.Ioo 0 T), t ≠ δ :=
    MeasureTheory.ae_restrict_of_ae hne0
  filter_upwards [MeasureTheory.ae_restrict_mem measurableSet_Ioo, hne] with t ht htδ
  exact hasDerivAt_rampTrajectory_of_ne δ hδ z A htδ (hz t ht)

end ModernODE
