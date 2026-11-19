import ModernODE.Theorem2ODE

/-!
# A concrete instance of Theorem 2: `ẋ = x` on `ℝ`, `x₀ = 1`, `T = 2`

Every hypothesis of `theorem2_ode` is discharged for `f(x,t) = x` with
`λ = a = b = 1` (so `λT = 2`), so the theorem is not vacuous:

* the transition operator is `Φ(t,τ) = e^{t-τ}` (forced by `isTransition_unique`),
  so `b e^{λ(t-τ)} ≤ ‖Φ(t,τ)‖ ≤ a e^{λ(t-τ)}` holds with equality;
* the exact solution is `s(t) = e^t`.
-/

noncomputable section

namespace ModernODE

open MeasureTheory Set Filter Topology

/-- The field `f(x,t) = x`, with `J = ∂ₓf = 1`. -/
def posField : C1Field ℝ where
  f x _ := x
  J _ _ := 1
  continuous_f := continuous_fst
  continuous_J := continuous_const
  hasFDerivAt x _ := hasFDerivAt_id x

def posTransition (t τ : ℝ) : ℝ →L[ℝ] ℝ := Real.exp (t - τ) • (1 : ℝ →L[ℝ] ℝ)

theorem isTransition_posTransition (s : ℝ → ℝ) (T : ℝ) :
    IsTransition (fun t => posField.J (s t) t) T posTransition := by
  refine ⟨Continuous.continuousOn (by unfold posTransition; fun_prop),
    fun τ _ => by simp [posTransition], fun τ t _ _ _ => ?_⟩
  have h : HasDerivAt (fun σ => posTransition σ τ)
      ((Real.exp (t - τ) * 1) • (1 : ℝ →L[ℝ] ℝ)) t := by
    unfold posTransition
    exact (((hasDerivAt_id t).sub_const τ).exp).smul_const (1 : ℝ →L[ℝ] ℝ)
  refine h.congr_deriv ?_
  ext
  simp [posTransition, posField]

theorem posExact_memLp :
    MemLp (fun t : ℝ => Real.exp t) 2 (volume.restrict (Ioc (0 : ℝ) 2)) :=
  memLp_of_continuousOn (Continuous.continuousOn (by fun_prop))

/-- `u = ṡ` for the exact solution `s(t) = e^t`. -/
def posExact : DerivCoord ℝ 2 := posExact_memLp.toLp _

theorem primFun_posExact {t : ℝ} (ht : t ∈ Icc 0 2) :
    primFun (DerivCoord.toL2 posExact) t = Real.exp t - 1 := by
  rw [primFun_congr (u := DerivCoord.toL2 posExact) posExact_memLp.coeFn_toLp ht]
  simp

theorem residual_posExact : residual posField 1 2 posExact = 0 := by
  unfold residual
  rw [sub_eq_zero]
  show posExact_memLp.toLp _ = _
  apply MemLp.toLp_congr
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  refine Eventually.of_forall fun t ht => ?_
  simp only [fieldAlong, traj, posField]
  rw [primFun_posExact (Ioc_subset_Icc_self ht)]
  ring

theorem posField_hbounds : ∀ Φ,
    IsTransition (fun t => posField.J (traj 1 (DerivCoord.toL2 posExact) t) t) 2 Φ →
    ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ 2 →
      1 * Real.exp (1 * (t - τ)) ≤ ‖Φ t τ‖ ∧ ‖Φ t τ‖ ≤ 1 * Real.exp (1 * (t - τ)) := by
  intro Φ hΦ τ t hτ hτt ht
  rw [isTransition_unique (continuousOn_jacAlong posField 1 _) hΦ
    (isTransition_posTransition _ 2) hτ hτt ht]
  have h : ‖posTransition t τ‖ = Real.exp (t - τ) := by simp [posTransition, norm_smul]
  rw [h]
  simp

/-- **Non-vacuity.** Theorem 2 (`theorem2_ode`) for `ẋ = x`, `x₀ = 1`, `T = 2`,
`λ = a = b = 1`, with every hypothesis discharged. -/
theorem theorem2_posField :
    sInf (Set.range (fun s => residualLoss (residual posField 1 2 s))) = 0 ∧
    Real.sqrt (optimalPL (residual posField 1 2) (meritVariation (residual posField 1 2))) ≤
      Real.exp 1 * 1 * 1 ^ 2 / 1 * Real.exp (-(1 * 2)) ∧
    ∀ μ, IsPL (residual posField 1 2) (meritVariation (residual posField 1 2)) μ →
      Real.sqrt μ ≤ Real.exp 1 * 1 * 1 ^ 2 / 1 * Real.exp (-(1 * 2)) :=
  theorem2_ode (T := 2) (lam := 1) (a := 1) (b := 1) posField 1 two_pos le_rfl one_pos one_pos
    (by norm_num) posExact residual_posExact posField_hbounds

end ModernODE
