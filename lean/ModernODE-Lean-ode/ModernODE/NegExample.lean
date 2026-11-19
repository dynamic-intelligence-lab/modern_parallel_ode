import ModernODE.Theorem1ODE

/-!
# A concrete instance: `ẋ = -x` on `ℝ`, `x₀ = 1`, `T = 1`

Every hypothesis of `theorem1_ode'` is discharged for `f(x,t) = -x` with
`a = c = 1`, so the theorem is not vacuous:

* the transition operator is `Φ(t,τ) = e^{-(t-τ)}` (forced by `isTransition_unique`);
* the exact solution is `s(t) = e^{-t}`.
-/

noncomputable section

namespace ModernODE

open MeasureTheory Set Filter Topology

/-- The field `f(x,t) = -x`, with `J = ∂ₓf = -1`. -/
def negField : C1Field ℝ where
  f x _ := -x
  J _ _ := -1
  continuous_f := continuous_fst.neg
  continuous_J := continuous_const
  hasFDerivAt x _ := (hasFDerivAt_id x).neg

/-! ### The transition operator `e^{-(t-τ)}` -/

def negTransition (t τ : ℝ) : ℝ →L[ℝ] ℝ := Real.exp (-(t - τ)) • (1 : ℝ →L[ℝ] ℝ)

theorem isTransition_negTransition (s : ℝ → ℝ) (T : ℝ) :
    IsTransition (fun t => negField.J (s t) t) T negTransition := by
  refine ⟨Continuous.continuousOn (by unfold negTransition; fun_prop),
    fun τ _ => by simp [negTransition], fun τ t _ _ _ => ?_⟩
  have h : HasDerivAt (fun σ => negTransition σ τ)
      ((Real.exp (-(t - τ)) * -1) • (1 : ℝ →L[ℝ] ℝ)) t := by
    unfold negTransition
    exact ((((hasDerivAt_id t).sub_const τ).neg).exp).smul_const (1 : ℝ →L[ℝ] ℝ)
  refine h.congr_deriv ?_
  ext
  simp [negTransition, negField]

theorem negField_hdecay : ∀ (u : DerivCoord ℝ 1) Φ,
    IsTransition (fun t => negField.J (traj 1 (DerivCoord.toL2 u) t) t) 1 Φ →
    ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ 1 → ‖Φ t τ‖ ≤ 1 * Real.exp (-1 * (t - τ)) := by
  intro u Φ hΦ τ t hτ hτt ht
  rw [isTransition_unique (continuousOn_jacAlong negField 1 _) hΦ
    (isTransition_negTransition _ 1) hτ hτt ht]
  simp [negTransition, norm_smul]

/-! ### The exact solution `s(t) = e^{-t}` -/

theorem negExact_memLp :
    MemLp (fun t : ℝ => -Real.exp (-t)) 2 (volume.restrict (Ioc (0 : ℝ) 1)) :=
  memLp_of_continuousOn (Continuous.continuousOn (by fun_prop))

/-- `u = ṡ` for `s(t) = e^{-t}`. -/
def negExact : DerivCoord ℝ 1 := negExact_memLp.toLp _

theorem primFun_negExact {t : ℝ} (ht : t ∈ Icc 0 1) :
    primFun (DerivCoord.toL2 negExact) t = Real.exp (-t) - 1 := by
  rw [primFun_congr (u := DerivCoord.toL2 negExact) negExact_memLp.coeFn_toLp ht]
  simp

theorem residual_negExact : residual negField 1 1 negExact = 0 := by
  unfold residual
  rw [sub_eq_zero]
  show negExact_memLp.toLp _ = _
  apply MemLp.toLp_congr
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  refine Eventually.of_forall fun t ht => ?_
  simp only [fieldAlong, traj, negField]
  rw [primFun_negExact (Ioc_subset_Icc_self ht)]
  ring

/-- **Non-vacuity.** Theorem 1 (`theorem1_ode'`) for `ẋ = -x`, `x₀ = 1`,
`T = 1`, `a = c = 1`, with every hypothesis discharged. -/
theorem theorem1_negField :
    sInf (Set.range (fun u => residualLoss (residual negField 1 1 u))) = 0 ∧
    IsPL (residual negField 1 1) (meritVariation (residual negField 1 1))
      (((1 / 1) / (1 - Real.exp (-1 * 1))) ^ 2) ∧
    (1 / 1) / (1 - Real.exp (-1 * 1)) ≤
      Real.sqrt (optimalPL (residual negField 1 1) (meritVariation (residual negField 1 1))) ∧
    (1 : ℝ) / 1 ≤ (1 / 1) / (1 - Real.exp (-1 * 1)) ∧ (0 : ℝ) < 1 / 1 :=
  theorem1_ode' (T := 1) (a := 1) (c := 1) negField 1 le_rfl one_pos one_pos negField_hdecay
    ⟨negExact, residual_negExact⟩

end ModernODE
