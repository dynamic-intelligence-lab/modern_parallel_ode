import ModernODE.Core
import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Actual directional derivative of the squared-residual loss

The residual need not be a bounded map in the trajectory L² norm. The
hypothesis is differentiability along an individual real-parameter path,
which is the relevant notion for the appendix's first variation.
-/

noncomputable section

namespace ModernODE

open scoped RealInnerProductSpace

variable {V H : Type*} [SeminormedAddCommGroup V] [NormedSpace ℝ V]
  [NormedAddCommGroup H] [InnerProductSpace ℝ H]

/-- The directional derivative of the actual squared-residual merit.
Use the accompanying theorems only when the indicated derivative exists. -/
def meritVariation (r : V → H) (s v : V) : ℝ :=
  deriv (fun ε : ℝ => residualLoss (r (s + ε • v))) 0

 theorem hasDerivAt_residualLoss {q : ℝ → H} {q' : H} {t : ℝ}
    (hq : HasDerivAt q q' t) :
    HasDerivAt (fun ε => residualLoss (q ε)) (inner ℝ (q t) q') t := by
  simpa [residualLoss] using hq.norm_sq.div_const 2

/-- The first variation formula is derived from the derivative of the
residual along a variation, not simply assigned as a definition. -/
theorem meritVariation_eq (r : V → H) (s v : V) (q' : H)
    (hq : HasDerivAt (fun ε : ℝ => r (s + ε • v)) q' 0) :
    meritVariation r s v = inner ℝ (r s) q' := by
  simpa [meritVariation] using (hasDerivAt_residualLoss hq).deriv

/-- The first variation in a residual correction direction equals twice
the loss. This supplies the duality identity used in the stable proof. -/
theorem meritVariation_correction (r : V → H) (s η : V)
    (hq : HasDerivAt (fun ε : ℝ => r (s + ε • η)) (r s) 0) :
    meritVariation r s η = ‖r s‖ ^ 2 := by
  rw [meritVariation_eq r s η (r s) hq, real_inner_self_eq_norm_sq]

omit [SeminormedAddCommGroup V] [NormedSpace ℝ V] [InnerProductSpace ℝ H] in
/-- A zero residual realizes the global minimum of the merit functional. -/
theorem residualLoss_isLeast (r : V → H) (x : V) (hx : r x = 0) :
    IsLeast (Set.range (fun s => residualLoss (r s))) 0 := by
  constructor
  · exact ⟨x, by simp [residualLoss, hx]⟩
  · rintro y ⟨s, rfl⟩
    exact div_nonneg (sq_nonneg _) (by norm_num)

omit [SeminormedAddCommGroup V] [NormedSpace ℝ V] [InnerProductSpace ℝ H] in
/-- Consequently the real infimum of the loss is zero whenever an exact
zero-residual trajectory is supplied. -/
theorem residualLoss_inf_eq_zero (r : V → H) (x : V) (hx : r x = 0) :
    sInf (Set.range (fun s => residualLoss (r s))) = 0 := by
  exact (residualLoss_isLeast r x hx).csInf_eq

end ModernODE
