import ModernODE.PosExample

/-!
# Two corollaries matching the manuscript's wording

* `C1Field.ofContDiff`: a field that is `C¹` jointly in `(x,t)` (the
  manuscript's "continuously differentiable") is a `C1Field`, so
  `theorem1_ode'` and `theorem2_ode` apply to it. `C1Field` itself does not
  need `∂ₜf`.
* `optimalPL_tendsto_zero`: the note in Theorem 2 that, for fixed `a`, `b`,
  `λ`, the optimal PL constant tends to zero as `T → ∞`. Hypothesis: for all
  large horizons the hypotheses of `theorem2_ode` hold with the same
  constants. `posField_optimalPL_tendsto_zero` discharges it for `ẋ = x`.
-/

noncomputable section

namespace ModernODE

open MeasureTheory Set Filter Topology

/-- A field that is `C¹` jointly in `(x,t)` is a `C1Field`, with
`J x t = ∂ₓf(x,t)`. -/
def C1Field.ofContDiff {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (f : E → ℝ → E) (hf : ContDiff ℝ 1 (fun p : E × ℝ => f p.1 p.2)) : C1Field E where
  f := f
  J x t := (fderiv ℝ (fun p : E × ℝ => f p.1 p.2) (x, t)).comp (ContinuousLinearMap.inl ℝ E ℝ)
  continuous_f := hf.continuous
  continuous_J := (hf.continuous_fderiv le_rfl).clm_comp continuous_const
  hasFDerivAt x t := by
    have h := (hf.differentiable le_rfl (x, t)).hasFDerivAt.comp x
      (hasFDerivAt_prodMk_left (𝕜 := ℝ) x t)
    exact h

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- **`μ⋆ → 0` as `T → ∞`** (the note in the manuscript's Theorem 2). If for
every large horizon `T` the IVP has a solution on `[0,T]` along which
`b e^{λ(t-τ)} ≤ ‖Φ(t,τ)‖ ≤ a e^{λ(t-τ)}` with the same `a`, `b`, `λ`, then the
optimal PL constant on `[0,T]` tends to zero. -/
theorem optimalPL_tendsto_zero [FiniteDimensional ℝ E] [Nontrivial E] (F : C1Field E)
    (x₀ : E) {lam a b : ℝ} (ha : 1 ≤ a) (hb : 0 < b) (hlam : 0 < lam)
    (hdata : ∀ᶠ T in atTop, ∃ u : DerivCoord E T, residual F x₀ T u = 0 ∧
      ∀ Φ, IsTransition (fun t => F.J (traj x₀ (DerivCoord.toL2 u) t) t) T Φ →
        ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
          b * Real.exp (lam * (t - τ)) ≤ ‖Φ t τ‖ ∧ ‖Φ t τ‖ ≤ a * Real.exp (lam * (t - τ))) :
    Tendsto (fun T => optimalPL (residual F x₀ T) (meritVariation (residual F x₀ T)))
      atTop (𝓝 0) := by
  refine unstable_pl_family_tendsto_zero a b lam hlam _
    (Eventually.of_forall fun T => Real.sSup_nonneg fun μ (hμ : IsPL _ _ μ) => hμ.1) ?_
  filter_upwards [hdata, eventually_ge_atTop (2 / lam)] with T ⟨u, hu, hbd⟩ hT
  have h2 : 2 ≤ lam * T := by
    rw [div_le_iff₀ hlam] at hT
    linarith
  have hT0 : 0 < T := by
    have := div_pos (two_pos : (0 : ℝ) < 2) hlam
    linarith
  exact (theorem2_ode F x₀ hT0 ha hb hlam h2 u hu hbd).2.1

/-! ## Non-vacuity: `ẋ = x`, `x₀ = 1`, every horizon `T` -/

theorem posExactT_memLp (T : ℝ) :
    MemLp (fun t : ℝ => Real.exp t) 2 (volume.restrict (Ioc (0 : ℝ) T)) :=
  memLp_of_continuousOn (T := T) (Continuous.continuousOn (by fun_prop))

/-- `u = ṡ` for the exact solution `s(t) = e^t` on `[0,T]`. -/
def posExactT (T : ℝ) : DerivCoord ℝ T := (posExactT_memLp T).toLp _

theorem residual_posExactT (T : ℝ) : residual posField 1 T (posExactT T) = 0 := by
  unfold residual
  rw [sub_eq_zero]
  show (posExactT_memLp T).toLp _ = _
  apply MemLp.toLp_congr
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  refine Eventually.of_forall fun t ht => ?_
  simp only [fieldAlong, traj, posField]
  rw [primFun_congr (u := DerivCoord.toL2 (posExactT T)) (posExactT_memLp T).coeFn_toLp
    (Ioc_subset_Icc_self ht)]
  simp

theorem posField_hboundsT (T : ℝ) : ∀ Φ,
    IsTransition (fun t => posField.J (traj 1 (DerivCoord.toL2 (posExactT T)) t) t) T Φ →
    ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
      1 * Real.exp (1 * (t - τ)) ≤ ‖Φ t τ‖ ∧ ‖Φ t τ‖ ≤ 1 * Real.exp (1 * (t - τ)) := by
  intro Φ hΦ τ t hτ hτt ht
  rw [isTransition_unique (continuousOn_jacAlong posField 1 _) hΦ
    (isTransition_posTransition _ T) hτ hτt ht]
  have h : ‖posTransition t τ‖ = Real.exp (t - τ) := by simp [posTransition, norm_smul]
  rw [h]
  simp

/-- **Non-vacuity** of `optimalPL_tendsto_zero`: for `ẋ = x`, `x₀ = 1`, the
optimal PL constant on `[0,T]` tends to zero as `T → ∞`. -/
theorem posField_optimalPL_tendsto_zero :
    Tendsto (fun T => optimalPL (residual posField 1 T) (meritVariation (residual posField 1 T)))
      atTop (𝓝 0) :=
  optimalPL_tendsto_zero (lam := 1) (a := 1) (b := 1) posField 1 le_rfl one_pos one_pos
    (Eventually.of_forall fun T => ⟨posExactT T, residual_posExactT T, posField_hboundsT T⟩)

end ModernODE
