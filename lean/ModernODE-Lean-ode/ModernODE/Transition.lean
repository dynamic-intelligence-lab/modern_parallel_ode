import ModernODE.Volterra

/-!
# Transition bounds imply pointwise Volterra domination

The variation-of-constants representation is an explicit analytic hypothesis.
The step from the actual operator bound to a scalar integral bound is proved.
-/

noncomputable section

namespace ModernODE

open MeasureTheory

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The causal, zero-extended state-transition action on a residual path. -/
def transitionAction (Φ : ℝ → ℝ → E →L[ℝ] E) (T : ℝ)
    (r : ℝ → E) (t τ : ℝ) : E :=
  if 0 ≤ τ ∧ τ ≤ t ∧ t ≤ T then Φ t τ (r τ) else 0

/-- Operator exponential decay bounds the norm of the causal integrand. -/
theorem norm_transitionAction_le
    (Φ : ℝ → ℝ → E →L[ℝ] E) (r : ℝ → E) (a c T : ℝ)
    (hΦ : ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
      ‖Φ t τ‖ ≤ a * Real.exp (-c * (t - τ))) (t τ : ℝ) :
    ‖transitionAction Φ T r t τ‖ ≤ volterraKernel a c T t τ * ‖r τ‖ := by
  unfold transitionAction volterraKernel
  split_ifs with h
  · exact ((Φ t τ).le_opNorm (r τ)).trans
      (mul_le_mul_of_nonneg_right (hΦ τ t h.1 h.2.1 h.2.2) (norm_nonneg _))
  · simp

/-- A variation-of-constants representation and the transition bound imply
pointwise domination by the exponential Volterra kernel. -/
theorem correction_dominated_by_volterra
    (Φ : ℝ → ℝ → E →L[ℝ] E) (r η : ℝ → E) (a c T : ℝ)
    (hΦ : ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
      ‖Φ t τ‖ ≤ a * Real.exp (-c * (t - τ)))
    (hη : ∀ t, η t = ∫ τ, transitionAction Φ T r t τ)
    (hkr : ∀ t, Integrable (fun τ => volterraKernel a c T t τ * ‖r τ‖)) :
    ∀ t, ‖η t‖ ≤ ∫ τ, volterraKernel a c T t τ * ‖r τ‖ := by
  intro t
  rw [hη t]
  exact norm_integral_le_of_norm_le (hkr t)
    (Filter.Eventually.of_forall (norm_transitionAction_le Φ r a c T hΦ t))

end ModernODE
