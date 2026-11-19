import ModernODE.ODEInstance

/-!
# Existence and uniqueness of the state-transition operator

For `A` continuous on `[0,T]` the Peano–Baker series
`Φ = Σₙ Φₙ`, `Φ₀ = I`, `Φₙ₊₁(t,τ) = ∫_τ^t A(σ) Φₙ(σ,τ) dσ`,
converges (`‖Φₙ(t,τ)‖ ≤ (M(t-τ))ⁿ/n!`) and satisfies the defining equations
taken by `theorem1_ode` (`IsTransition`). Gronwall's inequality shows that
these equations have only one solution, so the decay hypothesis of
`theorem1_ode'` (in `Theorem1ODE`) is a condition on *the* transition operator.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace ModernODE

open MeasureTheory Set Filter Topology
open scoped Nat

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- The defining equations of the state-transition operator of `ẋ = A(t) x`
on `[0,T]`: the three hypotheses on `Φ` taken by `theorem1_ode`. -/
def IsTransition (A : ℝ → E →L[ℝ] E) (T : ℝ) (Φ : ℝ → ℝ → E →L[ℝ] E) : Prop :=
  ContinuousOn (fun p : ℝ × ℝ => Φ p.1 p.2) (timeTriangle T) ∧
  (∀ τ ∈ Icc 0 T, Φ τ τ = 1) ∧
  (∀ τ t, 0 ≤ τ → τ < t → t < T → HasDerivAt (fun σ => Φ σ τ) ((A t).comp (Φ t τ)) t)

section PeanoBaker

variable (B : ℝ → E →L[ℝ] E)

/-- Terms of the Peano–Baker series. -/
def pbTerm : ℕ → ℝ → ℝ → E →L[ℝ] E
  | 0 => fun _ _ => 1
  | n + 1 => fun t τ => ∫ σ in τ..t, (B σ).comp (pbTerm n σ τ)

/-- The Peano–Baker series. -/
def pbSum (t τ : ℝ) : E →L[ℝ] E := ∑' n, pbTerm B n t τ

theorem pbSum_self (τ : ℝ) : pbSum B τ τ = 1 := by
  unfold pbSum
  rw [tsum_eq_single 0 fun n hn => ?_]
  · rfl
  · cases n with
    | zero => exact absurd rfl hn
    | succ m => exact intervalIntegral.integral_same

variable {B} (hB : Continuous B)
include hB

theorem continuous_pbTerm (n : ℕ) : Continuous fun p : ℝ × ℝ => pbTerm B n p.1 p.2 := by
  induction n with
  | zero => exact continuous_const
  | succ n ih =>
    have hg : Continuous (Function.uncurry
        fun (p : ℝ × ℝ) (σ : ℝ) => (B σ).comp (pbTerm B n σ p.2)) :=
      (hB.comp continuous_snd).clm_comp
        (ih.comp (continuous_snd.prodMk (continuous_snd.comp continuous_fst)))
    have h1 := intervalIntegral.continuous_parametric_intervalIntegral_of_continuous
      (a₀ := 0) (μ := volume) hg continuous_fst
    have h2 := intervalIntegral.continuous_parametric_intervalIntegral_of_continuous
      (a₀ := 0) (μ := volume) hg continuous_snd
    refine (h1.sub h2).congr fun p => ?_
    have hc : Continuous fun σ => (B σ).comp (pbTerm B n σ p.2) :=
      hg.comp (Continuous.prodMk_right p)
    exact intervalIntegral.integral_interval_sub_left (hc.intervalIntegrable _ _)
      (hc.intervalIntegrable _ _)

theorem continuous_pbIntegrand (n : ℕ) (τ : ℝ) :
    Continuous fun σ => (B σ).comp (pbTerm B n σ τ) :=
  hB.clm_comp ((continuous_pbTerm hB n).comp (continuous_id.prodMk continuous_const))

variable {M : ℝ} (hM0 : 0 ≤ M) (hM : ∀ σ, ‖B σ‖ ≤ M)
include hM0 hM

theorem norm_pbTerm_le (n : ℕ) {t τ : ℝ} (h : τ ≤ t) :
    ‖pbTerm B n t τ‖ ≤ (M * (t - τ)) ^ n / n ! := by
  induction n generalizing t with
  | zero =>
    simp only [pow_zero, Nat.factorial_zero, Nat.cast_one, div_one]
    exact ContinuousLinearMap.norm_id_le
  | succ n ih =>
    have hf : (n ! : ℝ) ≠ 0 := by positivity
    calc ‖pbTerm B (n + 1) t τ‖ = ‖∫ σ in τ..t, (B σ).comp (pbTerm B n σ τ)‖ := rfl
      _ ≤ ∫ σ in τ..t, M * ((M * (σ - τ)) ^ n / n !) := by
        refine intervalIntegral.norm_integral_le_of_norm_le h
          (Eventually.of_forall fun σ hσ => ?_) ?_
        · calc ‖(B σ).comp (pbTerm B n σ τ)‖ ≤ ‖B σ‖ * ‖pbTerm B n σ τ‖ :=
                ContinuousLinearMap.opNorm_comp_le _ _
            _ ≤ M * ((M * (σ - τ)) ^ n / n !) :=
                mul_le_mul (hM σ) (ih hσ.1.le) (norm_nonneg _) hM0
        · exact (by fun_prop : Continuous fun σ : ℝ => M * ((M * (σ - τ)) ^ n / n !)).intervalIntegrable _ _
      _ = (M * (t - τ)) ^ (n + 1) / (n + 1) ! := by
        have hfun : (fun σ : ℝ => M * ((M * (σ - τ)) ^ n / n !)) =
            fun σ => (M ^ (n + 1) / n !) * (σ - τ) ^ n := by
          funext σ
          rw [mul_pow]
          ring
        rw [hfun, intervalIntegral.integral_const_mul,
          intervalIntegral.integral_comp_sub_right (fun x => x ^ n) τ, sub_self, integral_pow,
          Nat.factorial_succ, mul_pow, zero_pow (Nat.succ_ne_zero n), sub_zero]
        set d := t - τ
        push_cast
        field_simp

theorem summable_pbTerm {t τ : ℝ} (h : τ ≤ t) : Summable fun n => pbTerm B n t τ :=
  Summable.of_norm_bounded (Real.summable_pow_div_factorial (M * (t - τ)))
    fun n => norm_pbTerm_le hB hM0 hM n h

/-- The Peano–Baker series solves the Volterra equation `Φ = I + ∫ A Φ`. -/
theorem pbSum_eq {t τ : ℝ} (h : τ ≤ t) :
    pbSum B t τ = 1 + ∫ σ in τ..t, (B σ).comp (pbSum B σ τ) := by
  have hint : HasSum (fun n => pbTerm B (n + 1) t τ)
      (∫ σ in τ..t, (B σ).comp (pbSum B σ τ)) := by
    refine intervalIntegral.hasSum_integral_of_dominated_convergence
      (fun n _ => M * ((M * (t - τ)) ^ n / n !))
      (fun n => (continuous_pbIntegrand hB n τ).aestronglyMeasurable)
      (fun n => Eventually.of_forall fun σ hσ => ?_)
      (Eventually.of_forall fun σ _ => (Real.summable_pow_div_factorial _).mul_left M)
      intervalIntegrable_const
      (Eventually.of_forall fun σ hσ => ?_)
    · rw [uIoc_of_le h] at hσ
      calc ‖(B σ).comp (pbTerm B n σ τ)‖ ≤ ‖B σ‖ * ‖pbTerm B n σ τ‖ :=
            ContinuousLinearMap.opNorm_comp_le _ _
        _ ≤ M * ((M * (t - τ)) ^ n / n !) := by
          refine mul_le_mul (hM σ) ((norm_pbTerm_le hB hM0 hM n hσ.1.le).trans ?_)
            (norm_nonneg _) hM0
          refine div_le_div_of_nonneg_right
            (pow_le_pow_left₀ (mul_nonneg hM0 (sub_nonneg.2 hσ.1.le)) ?_ n) (by positivity)
          exact mul_le_mul_of_nonneg_left (by linarith [hσ.2]) hM0
    · rw [uIoc_of_le h] at hσ
      simpa using ((summable_pbTerm hB hM0 hM hσ.1.le).hasSum).mapL
        (ContinuousLinearMap.compL ℝ E E E (B σ))
  show ∑' n, pbTerm B n t τ = _
  rw [(summable_pbTerm hB hM0 hM h).tsum_eq_zero_add, hint.tsum_eq]
  rfl

theorem continuousOn_pbSum (L : ℝ) :
    ContinuousOn (fun p : ℝ × ℝ => pbSum B p.1 p.2) {p | p.2 ≤ p.1 ∧ p.1 ≤ p.2 + L} := by
  refine continuousOn_tsum (fun n => (continuous_pbTerm hB n).continuousOn)
    (Real.summable_pow_div_factorial (M * L)) fun n p hp => ?_
  refine (norm_pbTerm_le hB hM0 hM n hp.1).trans ?_
  refine div_le_div_of_nonneg_right
    (pow_le_pow_left₀ (mul_nonneg hM0 (sub_nonneg.2 hp.1)) ?_ n) (by positivity)
  exact mul_le_mul_of_nonneg_left (by linarith [hp.2]) hM0

theorem hasDerivAt_pbSum {t τ : ℝ} (h : τ < t) :
    HasDerivAt (fun σ => pbSum B σ τ) ((B t).comp (pbSum B t τ)) t := by
  have hcont : ContinuousOn (fun σ => (B σ).comp (pbSum B σ τ)) (Icc τ (t + 1)) := by
    refine hB.continuousOn.clm_comp ?_
    exact (continuousOn_pbSum hB hM0 hM (t + 1 - τ)).comp
      (continuousOn_id.prodMk continuousOn_const)
      (fun σ hσ => ⟨hσ.1, show σ ≤ τ + (t + 1 - τ) by linarith [hσ.2]⟩)
  have hFTC : HasDerivAt (fun s => ∫ σ in τ..s, (B σ).comp (pbSum B σ τ))
      ((B t).comp (pbSum B t τ)) t :=
    intervalIntegral.integral_hasDerivAt_right
      ((hcont.mono (Icc_subset_Icc le_rfl (by linarith))).intervalIntegrable_of_Icc h.le)
      ((hcont.mono Ioo_subset_Icc_self).stronglyMeasurableAtFilter isOpen_Ioo t
        ⟨h, by linarith⟩)
      (hcont.continuousAt (Icc_mem_nhds h (by linarith)))
  refine (hFTC.const_add 1).congr_of_eventuallyEq ?_
  filter_upwards [Ioi_mem_nhds h] with s hs
  exact pbSum_eq hB hM0 hM (le_of_lt hs)

end PeanoBaker

/-- **Existence** of the state-transition operator on the whole interval. -/
theorem exists_isTransition {T : ℝ} (hT : 0 ≤ T) {A : ℝ → E →L[ℝ] E}
    (hA : ContinuousOn A (Icc 0 T)) : ∃ Φ, IsTransition A T Φ := by
  set B : ℝ → E →L[ℝ] E := fun t => A (projIcc 0 T hT t)
  have hB : Continuous B := hA.comp_continuous (continuous_subtype_val.comp continuous_projIcc) fun t => (projIcc 0 T hT t).2
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn hA
  have hM : ∀ σ, ‖B σ‖ ≤ max C 0 := fun σ =>
    (hC _ (projIcc 0 T hT σ).2).trans (le_max_left _ _)
  have hM0 : (0 : ℝ) ≤ max C 0 := le_max_right _ _
  refine ⟨pbSum B, ?_, fun τ _ => pbSum_self B τ, fun τ t hτ hτt htT => ?_⟩
  · exact (continuousOn_pbSum hB hM0 hM T).mono fun p hp =>
      ⟨hp.2.1, by linarith [hp.1, hp.2.2]⟩
  · have hBA : B t = A t := by
      simp only [B, projIcc_of_mem hT ⟨by linarith, htT.le⟩]
    rw [← hBA]
    exact hasDerivAt_pbSum hB hM0 hM hτt

/-- **Uniqueness**: the defining equations determine the transition operator
on the time triangle (Gronwall). -/
theorem isTransition_unique {T : ℝ} {A : ℝ → E →L[ℝ] E} (hA : ContinuousOn A (Icc 0 T))
    {Φ Ψ : ℝ → ℝ → E →L[ℝ] E} (hΦ : IsTransition A T Φ) (hΨ : IsTransition A T Ψ)
    {τ t : ℝ} (hτ : 0 ≤ τ) (hτt : τ ≤ t) (htT : t ≤ T) : Φ t τ = Ψ t τ := by
  rcases hτt.eq_or_lt with heq | hlt
  · subst heq
    exact (hΦ.2.1 τ ⟨hτ, htT⟩).trans (hΨ.2.1 τ ⟨hτ, htT⟩).symm
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn hA
  set K := max C 0
  set D : ℝ → E →L[ℝ] E := fun σ => Φ σ τ - Ψ σ τ
  have hmaps : MapsTo (fun σ : ℝ => (σ, τ)) (Icc τ T) (timeTriangle T) :=
    fun σ hσ => ⟨hτ, hσ.1, hσ.2⟩
  have hDcont : ContinuousOn D (Icc τ T) :=
    (hΦ.1.comp (continuousOn_id.prodMk continuousOn_const) hmaps).sub
      (hΨ.1.comp (continuousOn_id.prodMk continuousOn_const) hmaps)
  have hDτ : D τ = 0 := by
    simp only [D, hΦ.2.1 τ ⟨hτ, hlt.le.trans htT⟩, hΨ.2.1 τ ⟨hτ, hlt.le.trans htT⟩, sub_self]
  have hDderiv : ∀ σ ∈ Ioo τ T, HasDerivAt D ((A σ).comp (D σ)) σ := fun σ hσ => by
    have := (hΦ.2.2 τ σ hτ hσ.1 hσ.2).sub (hΨ.2.2 τ σ hτ hσ.1 hσ.2)
    simpa [D, ContinuousLinearMap.comp_sub] using this
  have hbound : ∀ σ ∈ Ioo τ T, ‖(A σ).comp (D σ)‖ ≤ K * ‖D σ‖ := fun σ hσ =>
    ((A σ).opNorm_comp_le _).trans (mul_le_mul_of_nonneg_right
      ((hC σ ⟨by linarith [hσ.1], hσ.2.le⟩).trans (le_max_left _ _)) (norm_nonneg _))
  have key : ∀ s ∈ Ioo τ t, ‖D t‖ ≤ ‖D s‖ * Real.exp (K * (t - τ)) := by
    intro s hs
    have h := norm_le_gronwallBound_of_norm_deriv_right_le (f := D)
      (f' := fun σ => (A σ).comp (D σ)) (δ := ‖D s‖) (K := K) (ε := 0)
      (hDcont.mono (Icc_subset_Icc hs.1.le htT))
      (fun σ hσ => (hDderiv σ ⟨hs.1.trans_le hσ.1, hσ.2.trans_le htT⟩).hasDerivWithinAt)
      le_rfl
      (fun σ hσ => by simpa using hbound σ ⟨hs.1.trans_le hσ.1, hσ.2.trans_le htT⟩)
      t ⟨hs.2.le, le_rfl⟩
    rw [gronwallBound_ε0] at h
    refine h.trans (mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_) (norm_nonneg _))
    exact mul_le_mul_of_nonneg_left (by linarith [hs.1]) (le_max_right _ _)
  have hc : ContinuousWithinAt D (Ioi τ) τ :=
    (hDcont τ ⟨le_rfl, hlt.le.trans htT⟩).mono_of_mem_nhdsWithin
      (Icc_mem_nhdsGT (hlt.trans_le htT))
  have hlim : Tendsto (fun s => ‖D s‖ * Real.exp (K * (t - τ))) (𝓝[>] τ) (𝓝 0) := by
    simpa [hDτ] using hc.tendsto.norm.mul_const (Real.exp (K * (t - τ)))
  have hle : ‖D t‖ ≤ 0 :=
    ge_of_tendsto hlim (by filter_upwards [Ioo_mem_nhdsGT hlt] with s hs using key s hs)
  exact sub_eq_zero.1 (norm_le_zero_iff.1 hle)

end ModernODE
