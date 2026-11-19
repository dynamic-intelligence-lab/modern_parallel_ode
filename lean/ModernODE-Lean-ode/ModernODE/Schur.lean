import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Tactic

/-!
# A scalar Schur estimate

The estimates here prove the integral Cauchy--Schwarz/Fubini part of the
stable PL argument. Integrability conditions are explicit; neither the
weighted Cauchy--Schwarz conclusion nor the L² gain is assumed.
-/

namespace ModernODE

open MeasureTheory

/-- Weighted Cauchy--Schwarz with a positive upper bound for the weight's mass.
The proof integrates a nonnegative square. -/
theorem weighted_integral_sq_le
    {X : Type*} [MeasurableSpace X] {ν : Measure X}
    (k r : X → ℝ) {B : ℝ} (hB : 0 < B)
    (hk_nonneg : ∀ x, 0 ≤ k x)
    (hk : Integrable k ν)
    (hkr : Integrable (fun x => k x * r x) ν)
    (hkr2 : Integrable (fun x => k x * (r x) ^ 2) ν)
    (hmass : (∫ x, k x ∂ν) ≤ B) :
    (∫ x, k x * r x ∂ν) ^ 2 ≤ B * ∫ x, k x * (r x) ^ 2 ∂ν := by
  let I := ∫ x, k x * r x ∂ν
  let q := I / B
  have hpoint : ∀ x, 2 * q * (k x * r x) ≤ k x * (r x) ^ 2 + q ^ 2 * k x := by
    intro x
    nlinarith [mul_nonneg (hk_nonneg x) (sq_nonneg (r x - q))]
  have hint := integral_mono (hkr.const_mul (2 * q)) (hkr2.add (hk.const_mul (q ^ 2))) hpoint
  change (∫ x, 2 * q * (k x * r x) ∂ν) ≤
    ∫ x, k x * (r x) ^ 2 + q ^ 2 * k x ∂ν at hint
  rw [integral_const_mul, integral_add hkr2 (hk.const_mul (q ^ 2)), integral_const_mul] at hint
  have hmass' := mul_le_mul_of_nonneg_left hmass (sq_nonneg q)
  have hidentity : 2 * q * I - q ^ 2 * B = I ^ 2 / B := by
    dsimp [q]
    field_simp
    ring
  have hdiv : I ^ 2 / B ≤ ∫ x, k x * (r x) ^ 2 ∂ν := by
    change 2 * q * I ≤ _ at hint
    linarith
  have h := (div_le_iff₀ hB).mp hdiv
  simpa [I, mul_comm] using h

/-- The L² Schur estimate for nonnegative kernels and scalar envelopes.
`η` and `r` can be pointwise norms of vector-valued trajectories. -/
theorem schur_energy_bound
    {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    {μ : Measure X} {ν : Measure Y} [SFinite μ] [SFinite ν]
    (k : X → Y → ℝ) (r : Y → ℝ) (η : X → ℝ) {B : ℝ}
    (hB : 0 < B)
    (hk_nonneg : ∀ x y, 0 ≤ k x y)
    (hr_nonneg : ∀ y, 0 ≤ r y)
    (hη_nonneg : ∀ x, 0 ≤ η x)
    (hrow : ∀ x, (∫ y, k x y ∂ν) ≤ B)
    (hcol : ∀ y, (∫ x, k x y ∂μ) ≤ B)
    (hdom : ∀ x, η x ≤ ∫ y, k x y * r y ∂ν)
    (hk : ∀ x, Integrable (k x) ν)
    (hkr : ∀ x, Integrable (fun y => k x y * r y) ν)
    (hkr2 : ∀ x, Integrable (fun y => k x y * (r y) ^ 2) ν)
    (hprod : Integrable (fun p : X × Y => k p.1 p.2 * (r p.2) ^ 2) (μ.prod ν))
    (hη2 : Integrable (fun x => (η x) ^ 2) μ)
    (hr2 : Integrable (fun y => (r y) ^ 2) ν) :
    (∫ x, (η x) ^ 2 ∂μ) ≤ B ^ 2 * ∫ y, (r y) ^ 2 ∂ν := by
  have hpoint : ∀ x, (η x) ^ 2 ≤ B * ∫ y, k x y * (r y) ^ 2 ∂ν := by
    intro x
    have hcs := weighted_integral_sq_le (k x) r hB (hk_nonneg x)
      (hk x) (hkr x) (hkr2 x) (hrow x)
    have hdomx := hdom x
    have hηx := hη_nonneg x
    have hIx : 0 ≤ ∫ y, k x y * r y ∂ν :=
      integral_nonneg fun y => mul_nonneg (hk_nonneg x y) (hr_nonneg y)
    have hsquare : (η x) ^ 2 ≤ (∫ y, k x y * r y ∂ν) ^ 2 :=
      (sq_le_sq₀ hηx hIx).2 hdomx
    exact hsquare.trans hcs
  have hfirst := integral_mono hη2 (hprod.integral_prod_left.const_mul B) hpoint
  rw [integral_const_mul, integral_integral_swap hprod] at hfirst
  have hsecond : (∫ y, ∫ x, k x y * (r y) ^ 2 ∂μ ∂ν) ≤ B * ∫ y, (r y) ^ 2 ∂ν := by
    calc
      _ ≤ ∫ y, B * (r y) ^ 2 ∂ν := by
        apply integral_mono hprod.integral_prod_right (hr2.const_mul B)
        intro y
        change (∫ x, k x y * (r y) ^ 2 ∂μ) ≤ B * (r y) ^ 2
        rw [integral_mul_const]
        exact mul_le_mul_of_nonneg_right (hcol y) (sq_nonneg (r y))
      _ = _ := integral_const_mul B _
  calc
    _ ≤ B * (∫ y, ∫ x, k x y * (r y) ^ 2 ∂μ ∂ν) := hfirst
    _ ≤ B * (B * ∫ y, (r y) ^ 2 ∂ν) := mul_le_mul_of_nonneg_left hsecond hB.le
    _ = _ := by ring

end ModernODE
