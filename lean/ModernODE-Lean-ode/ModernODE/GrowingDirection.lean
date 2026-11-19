import ModernODE.Unstable

/-!
# A single direction with persistent exponential growth

This file proves Step 1 of the manuscript's unstable-case proof for a
finite-dimensional family of continuous linear maps. Norm attainment is
proved from compactness of the unit sphere. The composition law and
transition bounds are explicit assumptions; no ODE construction is hidden
inside those assumptions.
-/

namespace ModernODE

/-- A continuous linear map with nontrivial finite-dimensional real domain
attains its operator norm at a unit vector. -/
theorem exists_unit_norm_attaining
    {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] [Nontrivial E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (L : E →L[ℝ] F) :
    ∃ v : E, ‖v‖ = 1 ∧ ‖L v‖ = ‖L‖ := by
  obtain ⟨w, hw⟩ := exists_norm_eq E (show (0 : ℝ) ≤ 1 by norm_num)
  have hs : (Metric.sphere (0 : E) 1).Nonempty := by
    exact ⟨w, by simpa using hw⟩
  obtain ⟨v, hv, hmax⟩ := (isCompact_sphere (0 : E) 1).exists_isMaxOn hs
    L.continuous.norm.continuousOn
  have hvnorm : ‖v‖ = 1 := by simpa using hv
  refine ⟨v, hvnorm, le_antisymm ?_ ?_⟩
  · simpa [hvnorm] using L.le_opNorm v
  · apply ContinuousLinearMap.opNorm_le_of_unit_norm (norm_nonneg (L v))
    intro x hx
    exact hmax (by simpa using hx)

/-- A final-time norm lower bound and forward norm upper bounds give one
unit initial direction which grows at every time in the interval. Only the
lower bound at `(T, 0)` is needed. This is the precise linear-algebraic
content of Step 1 in the proof of Theorem 2. -/
theorem exists_growing_direction
    {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] [Nontrivial E]
    (Φ : ℝ → ℝ → E →L[ℝ] E)
    (a b lam T : ℝ) (ha : 0 < a)
    (hcomp : ∀ t ∈ Set.Icc 0 T, (Φ T t).comp (Φ t 0) = Φ T 0)
    (hupper : ∀ τ t : ℝ, 0 ≤ τ → τ ≤ t → t ≤ T →
      ‖Φ t τ‖ ≤ a * Real.exp (lam * (t - τ)))
    (hlower : b * Real.exp (lam * T) ≤ ‖Φ T 0‖) :
    ∃ v : E, ‖v‖ = 1 ∧ ‖Φ T 0 v‖ = ‖Φ T 0‖ ∧
      ∀ t ∈ Set.Icc 0 T,
        b / a * Real.exp (lam * t) ≤ ‖Φ t 0 v‖ ∧
        ‖Φ t 0 v‖ ≤ a * Real.exp (lam * t) := by
  obtain ⟨v, hv, hnorm⟩ := exists_unit_norm_attaining (Φ T 0)
  refine ⟨v, hv, hnorm, ?_⟩
  intro t ht
  have hterminal : b * Real.exp (lam * T) ≤ ‖Φ T 0 v‖ := by
    simpa [hnorm] using hlower
  have hpropagate : ‖Φ T 0 v‖ ≤
      a * Real.exp (lam * (T - t)) * ‖Φ t 0 v‖ := by
    calc
      ‖Φ T 0 v‖ = ‖Φ T t (Φ t 0 v)‖ := by
        rw [← hcomp t ht]
        rfl
      _ ≤ ‖Φ T t‖ * ‖Φ t 0 v‖ := (Φ T t).le_opNorm (Φ t 0 v)
      _ ≤ a * Real.exp (lam * (T - t)) * ‖Φ t 0 v‖ :=
        mul_le_mul_of_nonneg_right (hupper t T ht.1 ht.2 le_rfl) (norm_nonneg _)
  constructor
  · exact intermediate_growth_lower_bound a b lam T t _ _ ha hterminal hpropagate
  · calc
      ‖Φ t 0 v‖ ≤ ‖Φ t 0‖ * ‖v‖ := (Φ t 0).le_opNorm v
      _ = ‖Φ t 0‖ := by rw [hv, mul_one]
      _ ≤ a * Real.exp (lam * t) := by
        simpa using hupper 0 t le_rfl ht.1 ht.2

end ModernODE
