import ModernODE.IntervalL2
import ModernODE.RampDerivative

/-!
# Adjoint duality from the actual backward differential equation

Integration by parts is proved by applying the fundamental theorem of
calculus on either side of the ramp corner. The adjoint solution is an
explicit hypothesis, but the integral and L² duality identities are proved.
-/

noncomputable section

open MeasureTheory Set
open scoped RealInnerProductSpace

namespace ModernODE

theorem integrable_inner_of_memLp_two
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {f g : ℝ → E} {μ : Measure ℝ} (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) :
    Integrable (fun t => inner ℝ (f t) (g t)) μ := by
  apply (L2.integrable_inner (𝕜 := ℝ) (hf.toLp f) (hg.toLp g)).congr
  filter_upwards [hf.coeFn_toLp, hg.coeFn_toLp] with t ht ht'
  rw [ht, ht']

theorem inner_toLp_eq_integral
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {f g : ℝ → E} {μ : Measure ℝ} (hf : MemLp f 2 μ) (hg : MemLp g 2 μ) :
    inner ℝ (hf.toLp f) (hg.toLp g) = ∫ t, inner ℝ (f t) (g t) ∂μ := by
  rw [L2.inner_def]
  apply integral_congr_ae
  filter_upwards [hf.coeFn_toLp, hg.coeFn_toLp] with t ht ht'
  rw [ht, ht']

/-- The actual forward and backward ODEs imply the adjoint identity used
in Step 3 of Theorem 2. No adjoint integral identity is assumed. -/
theorem ramp_adjoint_duality
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (δ T : ℝ) (z p : ℝ → E) (A : ℝ → E →L[ℝ] E)
    (hδ : 0 < δ) (hδT : δ < T)
    (hz : ContinuousOn z (Icc 0 T)) (hp : ContinuousOn p (Icc 0 T))
    (hzODE : ∀ t ∈ Ioo 0 T, HasDerivAt z (A t (z t)) t)
    (hpODE : ∀ t ∈ Ioo 0 T,
      HasDerivAt p (-(A t).adjoint (p t) - rampTrajectory δ z t) t)
    (hpT : p T = 0) :
    ‖rampTrajectoryL2 δ T z (hδ.le.trans hδT.le) hz‖ ^ 2 =
      inner ℝ ((continuousOn_memLp_two (hδ.le.trans hδT.le) hp).toLp p)
        (rampForcingL2 δ T z (hδ.le.trans hδT.le) hz) := by
  have hT : 0 ≤ T := hδ.le.trans hδT.le
  let η := rampTrajectory δ z
  let u := rampForcing δ z
  let q := fun t => inner ℝ (p t) (η t)
  let q' := fun t => inner ℝ (p t) (u t) - ‖η t‖ ^ 2
  have hη : ContinuousOn η (Icc 0 T) := continuousOn_rampTrajectory δ hz
  have hq : ContinuousOn q (Icc 0 T) := hp.inner hη
  have hp2 := continuousOn_memLp_two hT hp
  have hu2 := rampForcing_memLp_two δ hT hz
  have hiinner : IntervalIntegrable (fun t => inner ℝ (p t) (u t)) volume 0 T :=
    (intervalIntegrable_iff_integrableOn_Ioc_of_le hT).2
      (integrable_inner_of_memLp_two hp2 hu2)
  have hiη : IntervalIntegrable (fun t => ‖η t‖ ^ 2) volume 0 T :=
    (hη.norm.pow 2).intervalIntegrable_of_Icc hT
  have hi : IntervalIntegrable q' volume 0 T := hiinner.sub hiη
  have hid : IntervalIntegrable q' volume 0 δ := hi.mono_set (by
    simp only [uIcc_of_le hT, uIcc_of_le hδ.le]
    exact Icc_subset_Icc le_rfl hδT.le)
  have hit : IntervalIntegrable q' volume δ T := hi.mono_set (by
    simp only [uIcc_of_le hT, uIcc_of_le hδT.le]
    exact Icc_subset_Icc hδ.le le_rfl)
  have hd : ∀ t ∈ Ioo 0 T, t ≠ δ → HasDerivAt q (q' t) t := by
    intro t ht htδ
    have hηd := hasDerivAt_rampTrajectory_of_ne δ hδ z A htδ (hzODE t ht)
    have hqd := (hpODE t ht).inner ℝ hηd
    convert hqd using 1
    dsimp [q', η, u]
    rw [inner_add_right, inner_sub_left, inner_neg_left,
      ContinuousLinearMap.adjoint_inner_left, real_inner_self_eq_norm_sq]
    ring
  have hd0 : (∫ t in 0..δ, q' t) = q δ - q 0 :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hδ.le
      (hq.mono (Icc_subset_Icc le_rfl hδT.le))
      (fun t ht => hd t ⟨ht.1, ht.2.trans hδT⟩ ht.2.ne) hid
  have hdT : (∫ t in δ..T, q' t) = q T - q δ :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hδT.le
      (hq.mono (Icc_subset_Icc hδ.le le_rfl))
      (fun t ht => hd t ⟨hδ.trans ht.1, ht.2⟩ ht.1.ne') hit
  have hwhole : (∫ t in 0..T, q' t) = 0 := by
    rw [← intervalIntegral.integral_add_adjacent_intervals hid hit, hd0, hdT]
    have hbdy : q T - q 0 = 0 := by simp [q, η, rampTrajectory, hpT]
    linarith
  have henergy : trajectoryEnergy η T = ∫ t in 0..T, inner ℝ (p t) (u t) := by
    have hsub : (∫ t in 0..T, inner ℝ (p t) (u t)) -
        (∫ t in 0..T, ‖η t‖ ^ 2) = 0 := by
      rw [← intervalIntegral.integral_sub hiinner hiη]
      exact hwhole
    exact (sub_eq_zero.mp hsub).symm
  rw [norm_rampTrajectoryL2_sq]
  change trajectoryEnergy η T = inner ℝ (hp2.toLp p) (hu2.toLp u)
  rw [inner_toLp_eq_integral, henergy, intervalIntegral.integral_of_le hT]

end ModernODE
