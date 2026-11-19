import ModernODE.RampEnergy

/-!
# Concrete L² representatives of the ramp trajectory and forcing

This file connects the interval integrals in `RampEnergy` to the genuine
Mathlib Hilbert space `Lp E 2 (volume.restrict (Ioc 0 T))`.
-/

noncomputable section

open MeasureTheory Set
open scoped InnerProductSpace

namespace ModernODE

abbrev IntervalL2 (E : Type*) [NormedAddCommGroup E] (T : ℝ) :=
  Lp E 2 (volume.restrict (Ioc 0 T))

/-- Continuity on a compact interval implies square integrability there. -/
theorem continuousOn_memLp_two
    {E : Type*} [NormedAddCommGroup E] {f : ℝ → E} {T : ℝ}
    (hT : 0 ≤ T) (hf : ContinuousOn f (Icc 0 T)) :
    MemLp f 2 (volume.restrict (Ioc 0 T)) := by
  have hm : AEStronglyMeasurable f (volume.restrict (Ioc 0 T)) :=
    (hf.aestronglyMeasurable measurableSet_Icc).mono_measure
      (Measure.restrict_mono Ioc_subset_Icc_self le_rfl)
  exact (memLp_two_iff_integrable_sq_norm hm).2
    ((hf.norm.pow 2).intervalIntegrable_of_Icc hT).1

theorem continuousOn_rampTrajectory
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ : ℝ) {z : ℝ → E} {T : ℝ} (hz : ContinuousOn z (Icc 0 T)) :
    ContinuousOn (rampTrajectory δ z) (Icc 0 T) :=
  ((continuous_id.div_const δ).min continuous_const).continuousOn.smul hz

theorem rampForcing_memLp_two
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ : ℝ) {z : ℝ → E} {T : ℝ} (hT : 0 ≤ T)
    (hz : ContinuousOn z (Icc 0 T)) :
    MemLp (rampForcing δ z) 2 (volume.restrict (Ioc 0 T)) := by
  have hbase := (continuousOn_memLp_two hT hz).const_smul (δ⁻¹)
  have hi := MemLp.indicator (s := Iic δ) measurableSet_Iic hbase
  convert hi using 1

def rampTrajectoryL2
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ T : ℝ) (z : ℝ → E) (hT : 0 ≤ T) (hz : ContinuousOn z (Icc 0 T)) :
    IntervalL2 E T :=
  (continuousOn_memLp_two hT (continuousOn_rampTrajectory δ hz)).toLp
    (rampTrajectory δ z)

def rampForcingL2
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (δ T : ℝ) (z : ℝ → E) (hT : 0 ≤ T) (hz : ContinuousOn z (Icc 0 T)) :
    IntervalL2 E T :=
  (rampForcing_memLp_two δ hT hz).toLp (rampForcing δ z)

/-- The L² norm is the actual squared-norm interval integral. -/
theorem norm_toIntervalL2_sq
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {f : ℝ → E} {T : ℝ} (hT : 0 ≤ T)
    (hf : MemLp f 2 (volume.restrict (Ioc 0 T))) :
    ‖hf.toLp f‖ ^ 2 = trajectoryEnergy f T := by
  rw [← real_inner_self_eq_norm_sq, L2.inner_def]
  unfold trajectoryEnergy
  rw [intervalIntegral.integral_of_le hT]
  apply integral_congr_ae
  filter_upwards [hf.coeFn_toLp] with t ht
  rw [ht, real_inner_self_eq_norm_sq]

theorem norm_rampTrajectoryL2_sq
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (δ T : ℝ) (z : ℝ → E) (hT : 0 ≤ T) (hz : ContinuousOn z (Icc 0 T)) :
    ‖rampTrajectoryL2 δ T z hT hz‖ ^ 2 = trajectoryEnergy (rampTrajectory δ z) T :=
  norm_toIntervalL2_sq hT _

theorem norm_rampForcingL2_sq
    {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (δ T : ℝ) (z : ℝ → E) (hT : 0 ≤ T) (hz : ContinuousOn z (Icc 0 T)) :
    ‖rampForcingL2 δ T z hT hz‖ ^ 2 = trajectoryEnergy (rampForcing δ z) T :=
  norm_toIntervalL2_sq hT _

end ModernODE
