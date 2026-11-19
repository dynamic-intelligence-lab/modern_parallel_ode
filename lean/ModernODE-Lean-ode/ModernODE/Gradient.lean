import ModernODE.Core

/-!
# Agreement with the ordinary Hilbert gradient

When the first variation is represented by a Hilbert-space vector, the
extended-norm PL definition is exactly the usual squared-gradient inequality.
-/

namespace ModernODE

variable {S H K : Type*} [SeminormedAddCommGroup H]
  [NormedAddCommGroup K] [InnerProductSpace ℝ K]

/-- A finite directional bound on an inner-product functional is equivalent
to an upper bound on the norm of its representing gradient. -/
theorem boundsVariation_inner_iff (g : K) (G : ℝ) :
    BoundsVariation (fun v => inner ℝ g v) G ↔ ‖g‖ ≤ G := by
  constructor
  · intro hG
    by_cases hg : g = 0
    · simpa [hg] using hG.1
    · have hgpos : 0 < ‖g‖ := norm_pos_iff.mpr hg
      have h := hG.2 g
      dsimp only at h
      rw [real_inner_self_eq_norm_sq, abs_of_nonneg (sq_nonneg _)] at h
      apply (mul_le_mul_iff_left₀ hgpos).mp
      nlinarith only [h]
  · intro hG
    refine ⟨(norm_nonneg g).trans hG, ?_⟩
    intro v
    exact (abs_real_inner_le_norm g v).trans
      (mul_le_mul_of_nonneg_right hG (norm_nonneg v))

/-- On an ambient Hilbert space, a represented first variation recovers
exactly the familiar PL inequality, with no loss in the constant. -/
theorem isPL_inner_iff (r : S → H) (g : S → K) (μ : ℝ) :
    IsPL r (fun s v => inner ℝ (g s) v) μ ↔
      0 ≤ μ ∧ ∀ s, μ * ‖r s‖ ^ 2 ≤ ‖g s‖ ^ 2 := by
  constructor
  · rintro ⟨hμ, h⟩
    refine ⟨hμ, ?_⟩
    intro s
    exact h s ‖g s‖ ((boundsVariation_inner_iff _ _).2 le_rfl)
  · rintro ⟨hμ, h⟩
    refine ⟨hμ, ?_⟩
    intro s G hG
    have hg := (boundsVariation_inner_iff _ _).1 hG
    have hsq := (sq_le_sq₀ (norm_nonneg (g s)) hG.1).2 hg
    exact (h s).trans hsq

end ModernODE
