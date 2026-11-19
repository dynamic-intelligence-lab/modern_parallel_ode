import ModernODE.Theorem1ODE
import ModernODE.Theorem2

/-!
# Theorem 2 from the IVP

`theorem2` (in `Theorem2`) takes an abstract residual map, a transition
operator with the cocycle identity, and `UnstableAnalyticData`: a solution of
the backward adjoint equation and the forced nonlinear family. For the
concrete residual of `ODEInstance` all of these are proved here:

* the cocycle identity `Φ(T,t)Φ(t,0) = Φ(T,0)` (uniqueness, Gronwall);
* the adjoint solution `-ṗ - Aᵀp = η`, `p(T) = 0` (time reversal and the
  variation-of-constants path of `ODEInstance`);
* the forced IVP family `ṡ_ε = f(s_ε,t) + ε p(t)`, `s_ε(0) = x₀`, on all of
  `[0,T]` for small `ε` (Picard–Lindelöf for a truncated field, then
  Gronwall shows the truncation is inactive), with `‖s_ε - x‖∞ ≤ C|ε|`;
* its residual is exactly `ε p`, and integration by parts (Fubini, `p(T)=0`)
  gives the first-variation bound `‖ε g_ε‖ ‖∫v‖` with
  `g_ε = η + (A - J(s_ε))ᵀ p → η` in `L²`.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace ModernODE

open MeasureTheory Set Filter Topology Metric
open scoped NNReal

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-! ### Uniqueness for the linear ODE and the cocycle identity -/

/-- Gronwall: `D' = A D` on `(τ, T)`, `D` continuous on `[τ, T]` and `D(τ) = 0`
give `D = 0` on `[τ, T]`. -/
theorem linear_eq_zero_of_eq_zero {T τ : ℝ} {A : ℝ → E →L[ℝ] E}
    (hA : ContinuousOn A (Icc 0 T)) (hτ : 0 ≤ τ) {D : ℝ → E →L[ℝ] E}
    (hDcont : ContinuousOn D (Icc τ T)) (hDτ : D τ = 0)
    (hD : ∀ σ ∈ Ioo τ T, HasDerivAt D ((A σ).comp (D σ)) σ) {t : ℝ} (ht : t ∈ Icc τ T) :
    D t = 0 := by
  rcases ht.1.eq_or_lt with heq | hlt
  · rw [← heq]
    exact hDτ
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn hA
  set K := max C 0
  have hbound : ∀ σ ∈ Ioo τ T, ‖(A σ).comp (D σ)‖ ≤ K * ‖D σ‖ := fun σ hσ =>
    ((A σ).opNorm_comp_le _).trans (mul_le_mul_of_nonneg_right
      ((hC σ ⟨by linarith [hσ.1], hσ.2.le⟩).trans (le_max_left _ _)) (norm_nonneg _))
  have key : ∀ s ∈ Ioo τ t, ‖D t‖ ≤ ‖D s‖ * Real.exp (K * (t - τ)) := by
    intro s hs
    have h := norm_le_gronwallBound_of_norm_deriv_right_le (f := D)
      (f' := fun σ => (A σ).comp (D σ)) (δ := ‖D s‖) (K := K) (ε := 0)
      (hDcont.mono (Icc_subset_Icc hs.1.le ht.2))
      (fun σ hσ => (hD σ ⟨hs.1.trans_le hσ.1, hσ.2.trans_le ht.2⟩).hasDerivWithinAt)
      le_rfl
      (fun σ hσ => by simpa using hbound σ ⟨hs.1.trans_le hσ.1, hσ.2.trans_le ht.2⟩)
      t ⟨hs.2.le, le_rfl⟩
    rw [gronwallBound_ε0] at h
    refine h.trans (mul_le_mul_of_nonneg_left (Real.exp_le_exp.2 ?_) (norm_nonneg _))
    exact mul_le_mul_of_nonneg_left (by linarith [hs.1]) (le_max_right _ _)
  have hc : ContinuousWithinAt D (Ioi τ) τ :=
    (hDcont τ ⟨le_rfl, hlt.le.trans ht.2⟩).mono_of_mem_nhdsWithin
      (Icc_mem_nhdsGT (hlt.trans_le ht.2))
  have hlim : Tendsto (fun s => ‖D s‖ * Real.exp (K * (t - τ))) (𝓝[>] τ) (𝓝 0) := by
    simpa [hDτ] using hc.tendsto.norm.mul_const (Real.exp (K * (t - τ)))
  have hle : ‖D t‖ ≤ 0 :=
    ge_of_tendsto hlim (by filter_upwards [Ioo_mem_nhdsGT hlt] with s hs using key s hs)
  exact norm_le_zero_iff.1 hle

/-- The cocycle identity `Φ(σ,t) Φ(t,τ) = Φ(σ,τ)`. -/
theorem isTransition_comp {T : ℝ} {A : ℝ → E →L[ℝ] E} (hA : ContinuousOn A (Icc 0 T))
    {Φ : ℝ → ℝ → E →L[ℝ] E} (hΦ : IsTransition A T Φ) {τ t σ : ℝ} (hτ : 0 ≤ τ)
    (hτt : τ ≤ t) (htσ : t ≤ σ) (hσT : σ ≤ T) : (Φ σ t).comp (Φ t τ) = Φ σ τ := by
  have ht0 : 0 ≤ t := hτ.trans hτt
  have hmaps : ∀ {τ' : ℝ}, 0 ≤ τ' → τ' ≤ t →
      MapsTo (fun s : ℝ => (s, τ')) (Icc t T) (timeTriangle T) :=
    fun h0 h1 s hs => ⟨h0, h1.trans hs.1, hs.2⟩
  have hcont : ContinuousOn (fun s => (Φ s t).comp (Φ t τ) - Φ s τ) (Icc t T) :=
    ((hΦ.1.comp (continuousOn_id.prodMk continuousOn_const) (hmaps ht0 le_rfl)).clm_comp
      continuousOn_const).sub
      (hΦ.1.comp (continuousOn_id.prodMk continuousOn_const) (hmaps hτ hτt))
  have h0 : (Φ t t).comp (Φ t τ) - Φ t τ = 0 := by
    rw [hΦ.2.1 t ⟨ht0, htσ.trans hσT⟩]
    show (1 : E →L[ℝ] E) * Φ t τ - Φ t τ = 0
    rw [one_mul, sub_self]
  have hd : ∀ s ∈ Ioo t T, HasDerivAt (fun s => (Φ s t).comp (Φ t τ) - Φ s τ)
      ((A s).comp ((Φ s t).comp (Φ t τ) - Φ s τ)) s := by
    intro s hs
    have h1 := (hΦ.2.2 t s ht0 hs.1 hs.2).clm_comp (hasDerivAt_const s (Φ t τ))
    have h2 := hΦ.2.2 τ s hτ (hτt.trans_lt hs.1) hs.2
    convert h1.sub h2 using 1
    simp [ContinuousLinearMap.comp_sub, ContinuousLinearMap.comp_assoc]
  exact sub_eq_zero.1 (linear_eq_zero_of_eq_zero hA ht0 hcont h0 hd ⟨htσ, hσT⟩)

/-! ### The backward adjoint equation -/

/-- The backward adjoint equation `-ṗ - A(t)ᵀ p = η`, `p(T) = 0`, has a
solution: reverse time and use the variation-of-constants path. -/
theorem exists_adjoint_solution {T : ℝ} (hT : 0 ≤ T) {A : ℝ → E →L[ℝ] E}
    (hA : ContinuousOn A (Icc 0 T)) {η : ℝ → E} (hη : ContinuousOn η (Icc 0 T)) :
    ∃ p : ℝ → E, ContinuousOn p (Icc 0 T) ∧
      (∀ t ∈ Ioo 0 T, HasDerivAt p (-(A t).adjoint (p t) - η t) t) ∧ p T = 0 := by
  have hrev : MapsTo (fun s : ℝ => T - s) (Icc 0 T) (Icc 0 T) := fun s hs =>
    ⟨by linarith [hs.2], by linarith [hs.1]⟩
  set B : ℝ → E →L[ℝ] E := fun s => (A (T - s)).adjoint with hBdef
  set R : ℝ → E := fun s => η (T - s) with hRdef
  have hB : ContinuousOn B (Icc 0 T) :=
    ContinuousLinearMap.adjoint.continuous.comp_continuousOn
      (hA.comp (continuousOn_const.sub continuousOn_id) hrev)
  have hR : ContinuousOn R (Icc 0 T) := hη.comp (continuousOn_const.sub continuousOn_id) hrev
  have hRi : IntegrableOn R (Ioc 0 T) := hR.integrableOn_Icc.mono_set Ioc_subset_Icc_self
  obtain ⟨Ψ, hΨ⟩ := exists_isTransition hT hB
  set q := vocPath Ψ R with hqdef
  have hq : ContinuousOn q (Icc 0 T) := continuousOn_vocPath hB hΨ.1 hRi hΨ.2.1 hΨ.2.2
  have hg : ContinuousOn (fun σ => B σ (q σ) + R σ) (Icc 0 T) := (hB.clm_apply hq).add hR
  have hqeq : ∀ s ∈ Icc 0 T, q s = ∫ σ in (0 : ℝ)..s, (B σ (q σ) + R σ) := by
    intro s hs
    have h := voc_identity hB hΨ.1 hRi hΨ.2.1 hΨ.2.2 hs
    have hi1 : IntervalIntegrable (fun σ => B σ (q σ)) volume 0 s :=
      ((hB.clm_apply hq).mono (Icc_subset_Icc le_rfl hs.2)).intervalIntegrable_of_Icc hs.1
    have hi2 : IntervalIntegrable R volume 0 s :=
      (hR.mono (Icc_subset_Icc le_rfl hs.2)).intervalIntegrable_of_Icc hs.1
    rw [intervalIntegral.integral_add hi1 hi2, h]
    abel
  have hqd : ∀ s ∈ Ioo 0 T, HasDerivAt q (B s (q s) + R s) s := by
    intro s hs
    have hFTC := intervalIntegral.integral_hasDerivAt_right
      ((hg.mono (Icc_subset_Icc le_rfl hs.2.le)).intervalIntegrable_of_Icc hs.1.le)
      ((hg.mono Ioo_subset_Icc_self).stronglyMeasurableAtFilter isOpen_Ioo s hs)
      (hg.continuousAt (Icc_mem_nhds hs.1 hs.2))
    refine hFTC.congr_of_eventuallyEq ?_
    filter_upwards [Ioo_mem_nhds hs.1 hs.2] with σ hσ
    exact hqeq σ (Ioo_subset_Icc_self hσ)
  refine ⟨fun t => q (T - t), hq.comp (continuousOn_const.sub continuousOn_id) hrev,
    fun t ht => ?_, ?_⟩
  · have hTt : T - t ∈ Ioo 0 T := ⟨by linarith [ht.2], by linarith [ht.1]⟩
    have h : HasDerivAt (fun t => q (T - t))
        ((-1 : ℝ) • (B (T - t) (q (T - t)) + R (T - t))) t :=
      (hqd (T - t) hTt).scomp t ((hasDerivAt_id t).const_sub T)
    refine h.congr_deriv ?_
    simp only [hBdef, hRdef, sub_sub_cancel, neg_one_smul, neg_add_rev]
    abel
  · show q (T - T) = 0
    rw [sub_self]
    exact intervalIntegral.integral_same

/-! ### Integration by parts against a vector tail integral -/

/-- Fubini on the triangle: `∫₀ᵀ ⟨∫_t^T Q, v(t)⟩ dt = ∫₀ᵀ ⟨Q(σ), ∫₀^σ v⟩ dσ`. -/
theorem integral_inner_tail_eq {T : ℝ} {Q v : ℝ → E} (hQ : IntegrableOn Q (Ioc 0 T))
    (hv : IntegrableOn v (Ioc 0 T)) :
    ∫ t in Ioc 0 T, inner ℝ (∫ σ in t..T, Q σ) (v t) =
      ∫ σ in Ioc 0 T, inner ℝ (Q σ) (∫ t in (0 : ℝ)..σ, v t) := by
  set K : ℝ × ℝ → ℝ :=
    {p : ℝ × ℝ | p.2 ≤ p.1}.indicator fun p => inner ℝ (Q p.1) (v p.2) with hKdef
  have hK : Integrable K
      ((volume.restrict (Ioc (0 : ℝ) T)).prod (volume.restrict (Ioc (0 : ℝ) T))) := by
    refine Integrable.indicator ?_ (measurableSet_le measurable_snd measurable_fst)
    refine Integrable.mono' (Integrable.mul_prod hQ.norm hv.norm) ?_
      (Eventually.of_forall fun p => norm_inner_le_norm _ _)
    exact (hQ.aestronglyMeasurable.comp_quasiMeasurePreserving
      Measure.quasiMeasurePreserving_fst).inner
      (hv.aestronglyMeasurable.comp_quasiMeasurePreserving Measure.quasiMeasurePreserving_snd)
  have hin1 : ∀ σ ∈ Ioc (0 : ℝ) T,
      ∫ t in Ioc (0 : ℝ) T, K (σ, t) = inner ℝ (Q σ) (∫ t in (0 : ℝ)..σ, v t) := by
    intro σ hσ
    have : (fun t => K (σ, t)) = (Iic σ).indicator fun t => inner ℝ (Q σ) (v t) := by
      funext t
      simp only [hKdef, indicator, mem_setOf_eq, mem_Iic]
    rw [this, setIntegral_indicator measurableSet_Iic, Ioc_inter_Iic, inf_eq_right.2 hσ.2,
      integral_inner (hv.mono_set (Ioc_subset_Ioc_right hσ.2)) (Q σ),
      intervalIntegral.integral_of_le hσ.1.le]
  have hin2 : ∀ t ∈ Ioc (0 : ℝ) T,
      ∫ σ in Ioc (0 : ℝ) T, K (σ, t) = inner ℝ (∫ σ in t..T, Q σ) (v t) := by
    intro t ht
    have : (fun σ => K (σ, t)) = (Ici t).indicator fun σ => inner ℝ (v t) (Q σ) := by
      funext σ
      simp only [hKdef, indicator, mem_setOf_eq, mem_Ici, real_inner_comm]
    have hset : Ioc (0 : ℝ) T ∩ Ici t = Icc t T := by
      ext x
      simp only [mem_inter_iff, mem_Ioc, mem_Ici, mem_Icc]
      constructor
      · rintro ⟨⟨_, h2⟩, h3⟩
        exact ⟨h3, h2⟩
      · rintro ⟨h1, h2⟩
        exact ⟨⟨ht.1.trans_le h1, h2⟩, h1⟩
    rw [this, setIntegral_indicator measurableSet_Ici, hset, integral_Icc_eq_integral_Ioc,
      integral_inner (hQ.mono_set (Ioc_subset_Ioc_left ht.1.le)) (v t), real_inner_comm,
      intervalIntegral.integral_of_le ht.2]
  calc ∫ t in Ioc 0 T, inner ℝ (∫ σ in t..T, Q σ) (v t)
      = ∫ t in Ioc (0 : ℝ) T, ∫ σ in Ioc (0 : ℝ) T, K (σ, t) :=
        setIntegral_congr_fun measurableSet_Ioc fun t ht => (hin2 t ht).symm
    _ = ∫ σ in Ioc (0 : ℝ) T, ∫ t in Ioc (0 : ℝ) T, K (σ, t) :=
        (integral_integral_swap (f := fun σ t => K (σ, t)) hK).symm
    _ = ∫ σ in Ioc 0 T, inner ℝ (Q σ) (∫ t in (0 : ℝ)..σ, v t) :=
        setIntegral_congr_fun measurableSet_Ioc hin1

/-- `⟨∫_t^T Q, v⟩ = ⟨Q, ∫₀ᵗ v⟩` in `L²(0,T)`. -/
theorem inner_tail_eq_vec {T : ℝ} {Q : ℝ → E} (hQi : IntegrableOn Q (Ioc 0 T))
    (hφ : MemLp (fun t => ∫ σ in t..T, Q σ) 2 (volume.restrict (Ioc 0 T)))
    (hQ : MemLp Q 2 (volume.restrict (Ioc 0 T))) (v : IntervalL2 E T) :
    inner ℝ (hφ.toLp _ : IntervalL2 E T) v =
      inner ℝ (hQ.toLp _ : IntervalL2 E T) (primL2 T v) := by
  rw [MeasureTheory.L2.inner_def, MeasureTheory.L2.inner_def]
  have h1 : (fun t => inner ℝ ((hφ.toLp _ : IntervalL2 E T) t) (v t))
      =ᵐ[volume.restrict (Ioc 0 T)] fun t => inner ℝ (∫ σ in t..T, Q σ) (v t) := by
    filter_upwards [hφ.coeFn_toLp] with t ht
    rw [ht]
  have h2 : (fun σ => inner ℝ ((hQ.toLp _ : IntervalL2 E T) σ) ((primL2 T v : IntervalL2 E T) σ))
      =ᵐ[volume.restrict (Ioc 0 T)] fun σ => inner ℝ (Q σ) (∫ t in (0 : ℝ)..σ, v t) := by
    filter_upwards [hQ.coeFn_toLp, coeFn_primL2 v] with σ h1 h2
    rw [h1, h2]
    rfl
  rw [integral_congr_ae h1, integral_congr_ae h2]
  exact integral_inner_tail_eq hQi (integrableOn_Ioc_L2 v)

/-! ### The exact solution and candidates built from forced solutions -/

theorem ae_eq_fieldAlong_of_residual_eq_zero (F : C1Field E) (x₀ : E) {T : ℝ}
    {u : DerivCoord E T} (hu : residual F x₀ T u = 0) :
    (DerivCoord.toL2 u : ℝ → E) =ᵐ[volume.restrict (Ioc 0 T)]
      fieldAlong F x₀ (DerivCoord.toL2 u) := by
  have h' : DerivCoord.toL2 u =
      (memLp_of_continuousOn (continuousOn_fieldAlong F x₀ (DerivCoord.toL2 u))).toLp _ :=
    sub_eq_zero.1 hu
  have h2 := (memLp_of_continuousOn (continuousOn_fieldAlong F x₀ (DerivCoord.toL2 u))).coeFn_toLp
  rw [← h'] at h2
  exact h2

/-- The trajectory of an exact solution solves the IVP on `(0,T)`. -/
theorem hasDerivAt_traj_of_residual_eq_zero (F : C1Field E) (x₀ : E) {T : ℝ}
    {u : DerivCoord E T} (hu : residual F x₀ T u = 0) {t : ℝ} (ht : t ∈ Ioo 0 T) :
    HasDerivAt (traj x₀ (DerivCoord.toL2 u))
      (F.f (traj x₀ (DerivCoord.toL2 u) t) t) t := by
  set U := DerivCoord.toL2 u
  have hg := continuousOn_fieldAlong F x₀ U
  have heq : ∀ σ ∈ Icc 0 T, traj x₀ U σ = x₀ + ∫ τ in (0 : ℝ)..σ, fieldAlong F x₀ U τ :=
    fun σ hσ => by
      unfold traj
      rw [primFun_congr (ae_eq_fieldAlong_of_residual_eq_zero F x₀ hu) hσ]
  have hFTC := intervalIntegral.integral_hasDerivAt_right
    ((hg.mono (Icc_subset_Icc le_rfl ht.2.le)).intervalIntegrable_of_Icc ht.1.le)
    ((hg.mono Ioo_subset_Icc_self).stronglyMeasurableAtFilter isOpen_Ioo t ht)
    (hg.continuousAt (Icc_mem_nhds ht.1 ht.2))
  refine (hFTC.const_add x₀).congr_of_eventuallyEq ?_
  filter_upwards [Ioo_mem_nhds ht.1 ht.2] with σ hσ
  exact heq σ (Ioo_subset_Icc_self hσ)

theorem traj_zero (x₀ : E) {T : ℝ} (U : IntervalL2 E T) : traj x₀ U 0 = x₀ := by
  simp [traj, primFun]

/-- A solution of the forced IVP `ṡ = f(s,t) + ρ(t)`, `s(0) = x₀`, is a
candidate whose residual is exactly `ρ`. -/
theorem exists_candidate_of_forced (F : C1Field E) (x₀ : E) {T : ℝ} {s ρ : ℝ → E}
    (hs : ContinuousOn s (Icc 0 T)) (hs0 : s 0 = x₀) (hρ : ContinuousOn ρ (Icc 0 T))
    (hsd : ∀ t ∈ Ioo 0 T, HasDerivAt s (F.f (s t) t + ρ t) t) :
    ∃ v : DerivCoord E T, (∀ t ∈ Icc 0 T, traj x₀ (DerivCoord.toL2 v) t = s t) ∧
      residual F x₀ T v = (memLp_of_continuousOn hρ).toLp ρ := by
  have hg : ContinuousOn (fun t => F.f (s t) t + ρ t) (Icc 0 T) :=
    (F.continuous_f.comp_continuousOn (hs.prodMk continuousOn_id)).add hρ
  set v : DerivCoord E T := (memLp_of_continuousOn hg).toLp _ with hvdef
  have htraj : ∀ t ∈ Icc 0 T, traj x₀ (DerivCoord.toL2 v) t = s t := by
    intro t ht
    unfold traj
    rw [primFun_congr (u := DerivCoord.toL2 v) (memLp_of_continuousOn hg).coeFn_toLp ht,
      intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le ht.1
        (hs.mono (Icc_subset_Icc le_rfl ht.2)) (fun x hx => hsd x ⟨hx.1, hx.2.trans_le ht.2⟩)
        ((hg.mono (Icc_subset_Icc le_rfl ht.2)).intervalIntegrable_of_Icc ht.1), hs0]
    abel
  refine ⟨v, htraj, ?_⟩
  unfold residual
  show (memLp_of_continuousOn hg).toLp _ - _ = _
  rw [← MemLp.toLp_sub]
  apply MemLp.toLp_congr
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  refine Eventually.of_forall fun t ht => ?_
  simp only [Pi.sub_apply, fieldAlong, htraj t (Ioc_subset_Icc_self ht)]
  abel

/-! ### The forced IVP on all of `[0, T]` for small forcing -/

/-- For small `ε`, the forced IVP `ṡ = f(s,t) + ε p(t)`, `s(0) = y(0)`, has a
solution on `[0,T]` within `C|ε|` of the solution `y`. The deviation equation
is first truncated (`φ(w) w` with a smooth bump `φ`), so that Picard–Lindelöf
applies on the whole interval; Gronwall then shows the truncation is inactive. -/
theorem exists_forced_family [FiniteDimensional ℝ E] (F : C1Field E) {T : ℝ} (hT : 0 < T)
    {y : ℝ → E} (hy : ContinuousOn y (Icc 0 T))
    (hyd : ∀ t ∈ Ioo 0 T, HasDerivAt y (F.f (y t) t) t)
    {p : ℝ → E} (hp : ContinuousOn p (Icc 0 T)) :
    ∃ ε₀ > 0, ∃ C, ∀ ε : ℝ, |ε| < ε₀ → ∃ s : ℝ → E, s 0 = y 0 ∧ ContinuousOn s (Icc 0 T) ∧
      (∀ t ∈ Ioo 0 T, HasDerivAt s (F.f (s t) t + ε • p t) t) ∧
      ∀ t ∈ Icc 0 T, ‖s t - y t‖ ≤ C * |ε| := by
  -- The truncation `π w = φ(w) w`: the identity for `‖w‖ ≤ 1`, bounded by `2`, Lipschitz.
  let φ : ContDiffBump (0 : E) := ⟨1, 2, one_pos, one_lt_two⟩
  set π : E → E := fun w => φ w • w with hπdef
  have hπ_id : ∀ w, ‖w‖ ≤ 1 → π w = w := fun w hw => by
    have h1 : w ∈ closedBall (0 : E) φ.rIn := by
      show w ∈ closedBall (0 : E) 1
      rwa [mem_closedBall_zero_iff]
    simp only [hπdef, φ.one_of_mem_closedBall h1, one_smul]
  have hπ_le : ∀ w, ‖π w‖ ≤ ‖w‖ := fun w => by
    simp only [hπdef, norm_smul, Real.norm_eq_abs, abs_of_nonneg φ.nonneg]
    exact mul_le_of_le_one_left (norm_nonneg _) φ.le_one
  have hπ_two : ∀ w, ‖π w‖ ≤ 2 := fun w => by
    rcases lt_or_ge ‖w‖ 2 with h | h
    · exact (hπ_le w).trans h.le
    · have h0 : φ w = 0 := φ.zero_of_le_dist (by
        show (2 : ℝ) ≤ dist w 0
        rwa [dist_zero_right])
      simp only [hπdef, h0, zero_smul, norm_zero]
      norm_num
  obtain ⟨Lπ, hLπ⟩ : ∃ C, LipschitzWith C π :=
    ContDiff.lipschitzWith_of_hasCompactSupport (φ.hasCompactSupport.smul_right (f' := id))
      (φ.contDiff.smul contDiff_id) le_rfl
  -- `J` is bounded on the tube `{(t, y t + w) : t ∈ [0,T], ‖w‖ ≤ 2}`.
  have htube : IsCompact (Icc 0 T ×ˢ closedBall (0 : E) 2) :=
    isCompact_Icc.prod (isCompact_closedBall _ _)
  have hJc : ContinuousOn (fun q : ℝ × E => F.J (y q.1 + q.2) q.1)
      (Icc 0 T ×ˢ closedBall (0 : E) 2) :=
    F.continuous_J.comp_continuousOn
      (((hy.comp continuousOn_fst fun q hq => hq.1).add continuousOn_snd).prodMk continuousOn_fst)
  obtain ⟨K₀, hK₀⟩ := htube.exists_bound_of_continuousOn hJc
  set K := max K₀ 0 + 1 with hKdef
  have hK : 0 < K := by
    have := le_max_right K₀ 0
    linarith
  have hJK : ∀ t ∈ Icc 0 T, ∀ w ∈ closedBall (0 : E) 2, ‖F.J (y t + w) t‖ ≤ K :=
    fun t ht w hw => (hK₀ (t, w) ⟨ht, hw⟩).trans (by linarith [le_max_left K₀ 0])
  have hlip : ∀ t ∈ Icc 0 T,
      LipschitzOnWith ⟨K, hK.le⟩ (fun w => F.f (y t + w) t) (closedBall (0 : E) 2) := by
    intro t ht
    refine (convex_closedBall (0 : E) 2).lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
      (f' := fun w => F.J (y t + w) t) (fun w _ => ?_) fun w hw => ?_
    · have h := (F.hasFDerivAt (y t + w) t).comp w ((hasFDerivAt_id w).const_add (y t))
      simpa using h.hasFDerivWithinAt
    · rw [← NNReal.coe_le_coe, coe_nnnorm]
      exact hJK t ht w hw
  -- `p` is bounded.
  obtain ⟨P₀, hP₀⟩ := isCompact_Icc.exists_bound_of_continuousOn hp
  set P := max P₀ 0 with hPdef
  have hP0 : 0 ≤ P := le_max_right _ _
  have hP : ∀ t ∈ Icc 0 T, ‖p t‖ ≤ P := fun t ht => (hP₀ t ht).trans (le_max_left _ _)
  set C := P * (Real.exp (K * T) - 1) / K with hCdef
  have hC0 : 0 ≤ C := div_nonneg (mul_nonneg hP0
    (sub_nonneg.2 (Real.one_le_exp (mul_nonneg hK.le hT.le)))) hK.le
  refine ⟨min 1 (1 / (C + 1)), lt_min one_pos (div_pos one_pos (by linarith)), C,
    fun ε hε => ?_⟩
  have hε1 : |ε| ≤ 1 := hε.le.trans (min_le_left _ _)
  have hCε : C * |ε| < 1 := by
    have h1 : |ε| < 1 / (C + 1) := hε.trans_le (min_le_right _ _)
    calc C * |ε| ≤ C * (1 / (C + 1)) := mul_le_mul_of_nonneg_left h1.le hC0
      _ < 1 := by
        rw [mul_one_div, div_lt_one (by linarith)]
        linarith
  -- The truncated deviation field.
  set G : ℝ → E → E := fun t w => F.f (y t + π w) t - F.f (y t) t + ε • p t with hGdef
  have hGbound : ∀ t ∈ Icc 0 T, ∀ w, ‖G t w‖ ≤ K * ‖π w‖ + |ε| * P := by
    intro t ht w
    have h1 := (hlip t ht).dist_le_mul (π w) (mem_closedBall_zero_iff.2 (hπ_two w)) 0
      (mem_closedBall_self (by norm_num))
    simp only [add_zero, dist_eq_norm, sub_zero, NNReal.coe_mk] at h1
    have h2 := norm_add_le (F.f (y t + π w) t - F.f (y t) t) (ε • p t)
    have h3 : ‖ε • p t‖ ≤ |ε| * P := by
      rw [norm_smul, Real.norm_eq_abs]
      exact mul_le_mul_of_nonneg_left (hP t ht) (abs_nonneg ε)
    simp only [hGdef]
    linarith
  set L := 2 * K + P with hLdef
  have hL0 : 0 ≤ L := by linarith
  have hPL : IsPicardLindelof G (tmin := 0) (tmax := T) ⟨0, le_rfl, hT.le⟩ (0 : E)
      ⟨L * T, mul_nonneg hL0 hT.le⟩ 0 ⟨L, hL0⟩ (⟨K, hK.le⟩ * Lπ) := by
    refine ⟨fun t ht => ?_, fun w _ => ?_, fun t ht w _ => ?_, ?_⟩
    · have h1 : LipschitzOnWith (⟨K, hK.le⟩ * Lπ) (fun w => F.f (y t + π w) t) univ :=
        (hlip t ht).comp hLπ.lipschitzOnWith fun w _ => mem_closedBall_zero_iff.2 (hπ_two w)
      refine LipschitzOnWith.of_dist_le_mul fun w1 _ w2 _ => ?_
      have h2 := h1.dist_le_mul w1 (mem_univ _) w2 (mem_univ _)
      simp only [hGdef]
      rw [dist_add_right, dist_sub_right]
      exact h2
    · show ContinuousOn (fun t => F.f (y t + π w) t - F.f (y t) t + ε • p t) (Icc 0 T)
      exact ((F.continuous_f.comp_continuousOn ((hy.add continuousOn_const).prodMk
        continuousOn_id)).sub (F.continuous_f.comp_continuousOn (hy.prodMk continuousOn_id))).add
        (continuousOn_const.smul hp)
    · show ‖G t w‖ ≤ L
      have h1 := hGbound t ht w
      have h2 := mul_le_mul_of_nonneg_left (hπ_two w) hK.le
      have h3 := mul_le_mul_of_nonneg_right hε1 hP0
      linarith
    · show L * max (T - 0) (0 - 0) ≤ L * T - 0
      simp [hT.le]
  obtain ⟨w, hw0, hw⟩ := hPL.exists_eq_forall_mem_Icc_hasDerivWithinAt₀
  have hw0' : w 0 = 0 := hw0
  have hwc : ContinuousOn w (Icc 0 T) := fun t ht => (hw t ht).continuousWithinAt
  -- Gronwall: `‖w‖ ≤ C |ε| < 1`, so the truncation is inactive.
  have hgron : ∀ t ∈ Icc 0 T, ‖w t‖ ≤ C * |ε| := by
    intro t ht
    have h := norm_le_gronwallBound_of_norm_deriv_right_le (f := w)
      (f' := fun t => G t (w t)) (δ := 0) (K := K) (ε := |ε| * P) hwc
      (fun x hx => (hw x ⟨hx.1, hx.2.le⟩).mono_of_mem_nhdsWithin (Icc_mem_nhdsGE_of_mem hx))
      (by simp [hw0'])
      (fun x hx => (hGbound x ⟨hx.1, hx.2.le⟩ (w x)).trans
        (by linarith [mul_le_mul_of_nonneg_left (hπ_le (w x)) hK.le]))
      t ht
    simp only [gronwallBound_of_K_ne_0 hK.ne', zero_mul, zero_add, sub_zero] at h
    have hexp : Real.exp (K * t) ≤ Real.exp (K * T) :=
      Real.exp_le_exp.2 (mul_le_mul_of_nonneg_left ht.2 hK.le)
    have h2 : |ε| * P / K * (Real.exp (K * t) - 1) ≤ |ε| * P / K * (Real.exp (K * T) - 1) :=
      mul_le_mul_of_nonneg_left (by linarith) (div_nonneg (mul_nonneg (abs_nonneg _) hP0) hK.le)
    calc ‖w t‖ ≤ |ε| * P / K * (Real.exp (K * t) - 1) := h
      _ ≤ |ε| * P / K * (Real.exp (K * T) - 1) := h2
      _ = C * |ε| := by
        rw [hCdef]
        ring
  have hwsmall : ∀ t ∈ Icc 0 T, ‖w t‖ ≤ 1 := fun t ht => (hgron t ht).trans hCε.le
  refine ⟨fun t => y t + w t, by simp [hw0'], hy.add hwc, fun t ht => ?_, fun t ht => ?_⟩
  · have hwd : HasDerivAt w (G t (w t)) t :=
      (hw t (Ioo_subset_Icc_self ht)).hasDerivAt (Icc_mem_nhds ht.1 ht.2)
    have hGt : G t (w t) = F.f (y t + w t) t - F.f (y t) t + ε • p t := by
      simp only [hGdef, hπ_id (w t) (hwsmall t (Ioo_subset_Icc_self ht))]
    rw [hGt] at hwd
    show HasDerivAt (fun t => y t + w t) (F.f (y t + w t) t + ε • p t) t
    convert (hyd t ht).add hwd using 1
    abel
  · simp only [add_sub_cancel_left]
    exact hgron t ht

/-! ### The first variation along a forced candidate -/

/-- `g = η + (A - J(s))ᵀ p` along the candidate `v₀`. -/
def adjGap (F : C1Field E) (x₀ : E) {T : ℝ} (A : ℝ → E →L[ℝ] E) (η p : ℝ → E)
    (v₀ : DerivCoord E T) (t : ℝ) : E :=
  η t + (A t - F.J (traj x₀ (DerivCoord.toL2 v₀) t) t).adjoint (p t)

theorem continuousOn_adjGap (F : C1Field E) (x₀ : E) {T : ℝ} {A : ℝ → E →L[ℝ] E}
    (hA : ContinuousOn A (Icc 0 T)) {η p : ℝ → E} (hη : ContinuousOn η (Icc 0 T))
    (hp : ContinuousOn p (Icc 0 T)) (v₀ : DerivCoord E T) :
    ContinuousOn (adjGap F x₀ A η p v₀) (Icc 0 T) :=
  hη.add ((ContinuousLinearMap.adjoint.continuous.comp_continuousOn
    (hA.sub (continuousOn_jacAlong F x₀ (DerivCoord.toL2 v₀)))).clm_apply hp)

/-- If the residual is `ε p` with `p` solving the adjoint equation, the first
variation is `ε ⟨g, ∫₀ᵗ v⟩` with `g = η + (A - J(s))ᵀ p` (integration by
parts, `p(T) = 0`), hence bounded by `‖ε g‖ ‖v‖`. -/
theorem boundsVariation_of_adjoint (F : C1Field E) (x₀ : E) {T : ℝ} {A : ℝ → E →L[ℝ] E}
    (hA : ContinuousOn A (Icc 0 T)) {η p : ℝ → E} (hη : ContinuousOn η (Icc 0 T))
    (hp : ContinuousOn p (Icc 0 T))
    (hpd : ∀ t ∈ Ioo 0 T, HasDerivAt p (-(A t).adjoint (p t) - η t) t) (hpT : p T = 0)
    (v₀ : DerivCoord E T) (ε : ℝ)
    (hr : residual F x₀ T v₀ = ε • (memLp_of_continuousOn hp).toLp p) :
    BoundsVariation (meritVariation (residual F x₀ T) v₀)
      ‖ε • (memLp_of_continuousOn (continuousOn_adjGap F x₀ hA hη hp v₀)).toLp
        (adjGap F x₀ A η p v₀)‖ := by
  set Q : ℝ → E := fun t => (A t).adjoint (p t) + η t with hQdef
  have hQc : ContinuousOn Q (Icc 0 T) :=
    ((ContinuousLinearMap.adjoint.continuous.comp_continuousOn hA).clm_apply hp).add hη
  have hQi : IntegrableOn Q (Ioc 0 T) := hQc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
  -- `p` is the tail integral of `Q = -ṗ`.
  have htail : ∀ t ∈ Icc 0 T, (∫ σ in t..T, Q σ) = p t := by
    intro t ht
    have h := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le (f' := fun x => -Q x) ht.2
      (hp.mono (Icc_subset_Icc ht.1 le_rfl))
      (fun x hx => (hpd x ⟨ht.1.trans_lt hx.1, hx.2⟩).congr_deriv (by
        simp only [hQdef]
        abel))
      ((hQc.mono (Icc_subset_Icc ht.1 le_rfl)).neg.intervalIntegrable_of_Icc ht.2)
    rw [intervalIntegral.integral_neg, hpT, zero_sub] at h
    exact neg_inj.1 h
  have hφ : MemLp (fun t => ∫ σ in t..T, Q σ) 2 (volume.restrict (Ioc 0 T)) := by
    refine (memLp_of_continuousOn hp).ae_eq ?_
    rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
    exact Eventually.of_forall fun t ht => (htail t (Ioc_subset_Icc_self ht)).symm
  have hPeq : (memLp_of_continuousOn hp).toLp p = hφ.toLp _ := by
    apply MemLp.toLp_congr
    rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
    exact Eventually.of_forall fun t ht => (htail t (Ioc_subset_Icc_self ht)).symm
  have hJP : ContinuousOn
      (fun t => (F.J (traj x₀ (DerivCoord.toL2 v₀) t) t).adjoint (p t)) (Icc 0 T) :=
    (ContinuousLinearMap.adjoint.continuous.comp_continuousOn
      (continuousOn_jacAlong F x₀ _)).clm_apply hp
  refine ⟨norm_nonneg _, fun w => ?_⟩
  have hv : ‖w‖ = ‖primL2 T (DerivCoord.toL2 w)‖ := DerivCoord.norm_def w
  have h1 : inner ℝ ((memLp_of_continuousOn hp).toLp p : IntervalL2 E T) (DerivCoord.toL2 w) =
      inner ℝ ((memLp_of_continuousOn hQc).toLp Q : IntervalL2 E T)
        (primL2 T (DerivCoord.toL2 w)) := by
    rw [hPeq]
    exact inner_tail_eq_vec hQi hφ _ _
  have h2 : inner ℝ ((memLp_of_continuousOn hp).toLp p : IntervalL2 E T)
      (jacApplyL2 F x₀ (DerivCoord.toL2 v₀) (continuousOn_primFun (DerivCoord.toL2 w))) =
      inner ℝ ((memLp_of_continuousOn hJP).toLp _ : IntervalL2 E T)
        (primL2 T (DerivCoord.toL2 w)) := by
    unfold jacApplyL2
    rw [MeasureTheory.L2.inner_def, MeasureTheory.L2.inner_def]
    refine integral_congr_ae ?_
    filter_upwards [(memLp_of_continuousOn hp).coeFn_toLp,
      (memLp_of_continuousOn (continuousOn_jacApply F x₀ (DerivCoord.toL2 v₀)
        (continuousOn_primFun (DerivCoord.toL2 w)))).coeFn_toLp,
      (memLp_of_continuousOn hJP).coeFn_toLp, coeFn_primL2 (DerivCoord.toL2 w)]
      with t ht1 ht2 ht3 ht4
    rw [ht1, ht2, ht3, ht4]
    exact (ContinuousLinearMap.adjoint_inner_left _ _ _).symm
  have h3 : (memLp_of_continuousOn hQc).toLp Q - (memLp_of_continuousOn hJP).toLp _ =
      ((memLp_of_continuousOn (continuousOn_adjGap F x₀ hA hη hp v₀)).toLp
        (adjGap F x₀ A η p v₀) : IntervalL2 E T) := by
    rw [← MemLp.toLp_sub]
    apply MemLp.toLp_congr
    refine Eventually.of_forall fun t => ?_
    simp only [Pi.sub_apply, hQdef, adjGap, map_sub, ContinuousLinearMap.sub_apply]
    abel
  rw [meritVariation_eq _ _ _ _ (hasDerivAt_residual F x₀ v₀ w), hr, real_inner_smul_left,
    inner_sub_right, h1, h2, ← inner_sub_left, h3, hv, abs_mul, norm_smul, Real.norm_eq_abs,
    mul_assoc]
  exact mul_le_mul_of_nonneg_left (abs_real_inner_le_norm _ _) (abs_nonneg _)

/-! ### The nonlinear family of `UnstableAnalyticData` -/

/-- For a continuous `η`, the adjoint solution `p` and a family of forced
candidates `s_i` with residual `ε_i p`, first-variation bounds `‖ε_i g_i‖`,
`ε_i → 0`, and `g_i → η` in `L²`. -/
theorem exists_adjoint_family [FiniteDimensional ℝ E] (F : C1Field E) (x₀ : E) {T : ℝ}
    (hT : 0 < T) {u : DerivCoord E T} (hu : residual F x₀ T u = 0) {η : ℝ → E}
    (hη : ContinuousOn η (Icc 0 T)) :
    ∃ (p : ℝ → E) (hp : ContinuousOn p (Icc 0 T)),
      (∀ t ∈ Ioo 0 T, HasDerivAt p
        (-(F.J (traj x₀ (DerivCoord.toL2 u) t) t).adjoint (p t) - η t) t) ∧
      p T = 0 ∧
      ∃ (s : ℕ → DerivCoord E T) (ε : ℕ → ℝ) (g : ℕ → IntervalL2 E T),
        Tendsto ε atTop (𝓝 0) ∧ (∀ᶠ i in atTop, ε i ≠ 0) ∧
        (∀ᶠ i in atTop,
          residual F x₀ T (s i) = ε i • (continuousOn_memLp_two hT.le hp).toLp p) ∧
        (∀ᶠ i in atTop,
          BoundsVariation (meritVariation (residual F x₀ T) (s i)) ‖ε i • g i‖) ∧
        Tendsto g atTop (𝓝 ((continuousOn_memLp_two hT.le hη).toLp η)) := by
  set U := DerivCoord.toL2 u with hUdef
  set A : ℝ → E →L[ℝ] E := fun t => F.J (traj x₀ U t) t with hAdef
  have hA : ContinuousOn A (Icc 0 T) := continuousOn_jacAlong F x₀ U
  obtain ⟨p, hp, hpd, hpT⟩ := exists_adjoint_solution hT.le hA hη
  obtain ⟨ε₀, hε₀, C, hfam⟩ := exists_forced_family F hT (continuousOn_traj x₀ U)
    (fun t ht => hasDerivAt_traj_of_residual_eq_zero F x₀ hu ht) hp
  set ε : ℕ → ℝ := fun i => ε₀ / 2 * (1 / ((i : ℝ) + 1)) with hεdef
  have hεpos : ∀ i, 0 < ε i := fun i =>
    mul_pos (half_pos hε₀) (one_div_pos.2 (by positivity))
  have hεlt : ∀ i, |ε i| < ε₀ := fun i => by
    rw [abs_of_pos (hεpos i)]
    have h1 : 1 / ((i : ℝ) + 1) ≤ 1 := by
      rw [div_le_one (by positivity)]
      linarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)]
    calc ε i ≤ ε₀ / 2 * 1 := mul_le_mul_of_nonneg_left h1 (half_pos hε₀).le
      _ < ε₀ := by linarith
  have hεlim : Tendsto ε atTop (𝓝 0) := by
    have h := tendsto_one_div_add_atTop_nhds_zero_nat.const_mul (ε₀ / 2)
    rw [mul_zero] at h
    exact h
  choose S hS0 hSc hSd hSclose using fun i => hfam (ε i) (hεlt i)
  have hρc : ∀ i, ContinuousOn (fun t => ε i • p t) (Icc 0 T) := fun i =>
    continuousOn_const.smul hp
  choose V hVtraj hVres using fun i => exists_candidate_of_forced F x₀ (hSc i)
    (by rw [hS0 i]; exact traj_zero x₀ U) (hρc i) (hSd i)
  have hres : ∀ i, residual F x₀ T (V i) = ε i • (continuousOn_memLp_two hT.le hp).toLp p :=
    fun i => by
      rw [hVres i]
      exact MemLp.toLp_const_smul (ε i) (continuousOn_memLp_two hT.le hp)
  refine ⟨p, hp, hpd, hpT, V, ε, fun i => (memLp_of_continuousOn
      (continuousOn_adjGap F x₀ hA hη hp (V i))).toLp (adjGap F x₀ A η p (V i)),
    hεlim, Eventually.of_forall fun i => (hεpos i).ne', Eventually.of_forall hres,
    Eventually.of_forall fun i => boundsVariation_of_adjoint F x₀ hA hη hp hpd hpT (V i) (ε i)
      (hres i), ?_⟩
  -- `g_i → η`: `s_i → x` uniformly and `J` is uniformly continuous near the solution.
  rw [Metric.tendsto_atTop]
  intro δ hδ
  set Cμ : ℝ := (measureUnivNNReal (volume.restrict (Ioc (0 : ℝ) T)) : ℝ) ^
    ((2 : ENNReal).toReal⁻¹) with hCμdef
  have hCμ : 0 ≤ Cμ := by positivity
  obtain ⟨P₀, hP₀⟩ := isCompact_Icc.exists_bound_of_continuousOn hp
  set P := max P₀ 0 with hPdef
  have hP0 : 0 ≤ P := le_max_right _ _
  set δ' := δ / (2 * (Cμ * P + 1)) with hδ'def
  have hδ' : 0 < δ' := div_pos hδ (by positivity)
  have hK : IsCompact (Icc 0 T ×ˢ closedBall (0 : E) 1) :=
    isCompact_Icc.prod (isCompact_closedBall _ _)
  have hJc : ContinuousOn (fun q : ℝ × E => F.J (traj x₀ U q.1 + q.2) q.1)
      (Icc 0 T ×ˢ closedBall (0 : E) 1) :=
    F.continuous_J.comp_continuousOn ((((continuousOn_traj x₀ U).comp continuousOn_fst
      fun q hq => hq.1).add continuousOn_snd).prodMk continuousOn_fst)
  obtain ⟨ρ, hρ, hρJ⟩ :=
    Metric.uniformContinuousOn_iff.1 (hK.uniformContinuousOn_of_continuous hJc) δ' hδ'
  have hev : ∀ᶠ i in atTop, C * |ε i| < min ρ 1 := by
    have h : Tendsto (fun i => C * |ε i|) atTop (𝓝 0) := by
      simpa using (hεlim.abs).const_mul C
    exact h.eventually (gt_mem_nhds (lt_min hρ one_pos))
  obtain ⟨N, hN⟩ := eventually_atTop.1 hev
  refine ⟨N, fun i hi => ?_⟩
  rw [dist_eq_norm]
  have hgap : ContinuousOn
      (fun t => (A t - F.J (traj x₀ (DerivCoord.toL2 (V i)) t) t).adjoint (p t)) (Icc 0 T) :=
    (ContinuousLinearMap.adjoint.continuous.comp_continuousOn
      (hA.sub (continuousOn_jacAlong F x₀ _))).clm_apply hp
  have heq : (memLp_of_continuousOn (continuousOn_adjGap F x₀ hA hη hp (V i))).toLp
      (adjGap F x₀ A η p (V i)) - (continuousOn_memLp_two hT.le hη).toLp η =
      (memLp_of_continuousOn hgap).toLp _ := by
    rw [← MemLp.toLp_sub]
    apply MemLp.toLp_congr
    exact Eventually.of_forall fun t => by simp [adjGap]
  rw [heq]
  have hbd : ∀ᵐ t ∂(volume.restrict (Ioc (0 : ℝ) T)),
      ‖((memLp_of_continuousOn hgap).toLp _ : IntervalL2 E T) t‖ ≤ δ' * P := by
    filter_upwards [(memLp_of_continuousOn hgap).coeFn_toLp, ae_restrict_mem measurableSet_Ioc]
      with t h1 h2
    rw [h1]
    have ht : t ∈ Icc 0 T := Ioc_subset_Icc_self h2
    have hwt : ‖S i t - traj x₀ U t‖ < min ρ 1 := (hSclose i t ht).trans_lt (hN i hi)
    have hmem1 : (t, S i t - traj x₀ U t) ∈ Icc 0 T ×ˢ closedBall (0 : E) 1 :=
      ⟨ht, mem_closedBall_zero_iff.2 (hwt.le.trans (min_le_right _ _))⟩
    have hmem0 : (t, (0 : E)) ∈ Icc 0 T ×ˢ closedBall (0 : E) 1 :=
      ⟨ht, mem_closedBall_self zero_le_one⟩
    have hdist : dist (t, S i t - traj x₀ U t) (t, (0 : E)) < ρ := by
      rw [Prod.dist_eq, dist_self, dist_zero_right]
      exact (max_le (norm_nonneg _) le_rfl).trans_lt (hwt.trans_le (min_le_left _ _))
    have hJ := hρJ _ hmem1 _ hmem0 hdist
    simp only [add_sub_cancel, add_zero, dist_eq_norm] at hJ
    rw [hVtraj i t ht]
    calc ‖(A t - F.J (S i t) t).adjoint (p t)‖
        ≤ ‖(A t - F.J (S i t) t).adjoint‖ * ‖p t‖ := ContinuousLinearMap.le_opNorm _ _
      _ = ‖A t - F.J (S i t) t‖ * ‖p t‖ := by rw [LinearIsometryEquiv.norm_map]
      _ ≤ δ' * P := by
        refine mul_le_mul ?_ ((hP₀ t ht).trans (le_max_left _ _)) (norm_nonneg _) hδ'.le
        rw [← norm_neg, neg_sub]
        exact hJ.le
  calc ‖((memLp_of_continuousOn hgap).toLp _ : IntervalL2 E T)‖ ≤ Cμ * (δ' * P) :=
        Lp.norm_le_of_ae_bound (mul_nonneg hδ'.le hP0) hbd
    _ < δ := by
      rw [hδ'def]
      have h1 : 0 < Cμ * P + 1 := by positivity
      rw [show Cμ * (δ / (2 * (Cμ * P + 1)) * P) = δ * (Cμ * P / (2 * (Cμ * P + 1))) by ring]
      have h2 : Cμ * P / (2 * (Cμ * P + 1)) < 1 := by
        rw [div_lt_one (by positivity)]
        nlinarith [mul_nonneg hCμ hP0]
      calc δ * (Cμ * P / (2 * (Cμ * P + 1))) < δ * 1 := mul_lt_mul_of_pos_left h2 hδ
        _ = δ := mul_one δ

/-! ### Theorem 2 with the manuscript's hypotheses only -/

/-- **Theorem 2 from the IVP.** Let `u` be an exact solution of the IVP (its
trajectory `x = x₀ + ∫₀ᵗ u`), and let the state-transition operator `Φ_x` of
the linearization along `x` satisfy `b e^{λ(t-τ)} ≤ ‖Φ_x(t,τ)‖ ≤ a e^{λ(t-τ)}`
for `0 ≤ τ ≤ t ≤ T`, with `a ≥ 1`, `b > 0`, `λ > 0`, `λT ≥ 2`. Then
`√μ⋆ ≤ e λ a² / b · e^{-λT}`, and every PL constant `μ` obeys `√μ ≤ e λ a² / b · e^{-λT}`
(so the real supremum `optimalPL` is over a bounded set). By `exists_isTransition`
and `isTransition_unique` the bound is a condition on the unique `Φ_x`. -/
theorem theorem2_ode [FiniteDimensional ℝ E] [Nontrivial E] (F : C1Field E) (x₀ : E)
    {T lam a b : ℝ} (hT : 0 < T) (ha : 1 ≤ a) (hb : 0 < b) (hlam : 0 < lam)
    (horizon : 2 ≤ lam * T) (u : DerivCoord E T) (hu : residual F x₀ T u = 0)
    (hbounds : ∀ Φ, IsTransition (fun t => F.J (traj x₀ (DerivCoord.toL2 u) t) t) T Φ →
      ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T →
        b * Real.exp (lam * (t - τ)) ≤ ‖Φ t τ‖ ∧ ‖Φ t τ‖ ≤ a * Real.exp (lam * (t - τ))) :
    sInf (Set.range (fun s => residualLoss (residual F x₀ T s))) = 0 ∧
    Real.sqrt (optimalPL (residual F x₀ T) (meritVariation (residual F x₀ T))) ≤
      Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)) ∧
    ∀ μ, IsPL (residual F x₀ T) (meritVariation (residual F x₀ T)) μ →
      Real.sqrt μ ≤ Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)) := by
  set A : ℝ → E →L[ℝ] E := fun t => F.J (traj x₀ (DerivCoord.toL2 u) t) t with hAdef
  have hA : ContinuousOn A (Icc 0 T) := continuousOn_jacAlong F x₀ _
  obtain ⟨Φ, hΦ⟩ := exists_isTransition hT.le hA
  have hB := hbounds Φ hΦ
  have hcont : ∀ v : E, ContinuousOn (fun t => Φ t 0 v) (Icc 0 T) := fun v =>
    (hΦ.1.comp (continuousOn_id.prodMk continuousOn_const)
      fun t ht => ⟨le_rfl, ht.1, ht.2⟩).clm_apply continuousOn_const
  let D : UnstableAnalyticData T lam hT.le (residual F x₀ T) Φ :=
    { merit_differentiable := merit_differentiable_residual F x₀
      continuous := hcont
      linearizedField := A
      homogeneous := fun v t ht => by
        have h := (hΦ.2.2 0 t le_rfl ht.1 ht.2).clm_apply (hasDerivAt_const t v)
        simpa using h
      zero_residual := ⟨u, hu⟩
      adjoint_family := fun v _ =>
        exists_adjoint_family F x₀ hT hu (continuousOn_rampTrajectory (1 / lam) (hcont v)) }
  have hcomp : ∀ t ∈ Icc 0 T, (Φ T t).comp (Φ t 0) = Φ T 0 :=
    fun t ht => isTransition_comp hA hΦ le_rfl ht.1 ht.2 le_rfl
  have hupper : ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T → ‖Φ t τ‖ ≤ a * Real.exp (lam * (t - τ)) :=
    fun τ t h1 h2 h3 => (hB τ t h1 h2 h3).2
  have hlower : b * Real.exp (lam * T) ≤ ‖Φ T 0‖ := by
    simpa using (hB 0 T le_rfl hT.le le_rfl).1
  have hmain := theorem2 T lam a b hT (residual F x₀ T) Φ ha hb hlam horizon D hcomp hupper
    hlower
  -- The set of PL constants is bounded: one forced candidate has a nonzero residual
  -- and a finite first-variation bound.
  have hbdd : BddAbove {μ | IsPL (residual F x₀ T) (meritVariation (residual F x₀ T)) μ} := by
    have ha0 : 0 < a := by linarith
    obtain ⟨v, -, -, hgrowth⟩ :=
      exists_growing_direction Φ a b lam T ha0 hcomp hupper hlower
    have hδ : 0 < 1 / lam := by positivity
    have hδT : 1 / lam < T := by
      apply (div_lt_iff₀ hlam).2
      nlinarith
    have henergy := ramp_energy_bounds (fun t => Φ t 0 v) a b lam (1 / lam) T
      ha0 hb hlam hδ hδT (hcont v) hgrowth
    have hEpos : 0 < ‖rampTrajectoryL2 (1 / lam) T (fun t => Φ t 0 v) hT.le (hcont v)‖ ^ 2 := by
      rw [norm_rampTrajectoryL2_sq]
      refine lt_of_lt_of_le ?_ henergy.2
      have hcancel : 2 * lam * (1 / lam) = 2 := by field_simp
      rw [hcancel]
      have h2 : Real.exp 2 < Real.exp (2 * lam * T) := Real.exp_lt_exp.2 (by linarith)
      exact mul_pos (by positivity) (sub_pos.2 h2)
    have hηne : rampTrajectoryL2 (1 / lam) T (fun t => Φ t 0 v) hT.le (hcont v) ≠ 0 := by
      intro h0
      rw [h0, norm_zero] at hEpos
      norm_num at hEpos
    obtain ⟨p, hp, hpd, hpT, s, ε, g, -, hε, hr, hG, -⟩ :=
      exists_adjoint_family F x₀ hT hu (continuousOn_rampTrajectory (1 / lam) (hcont v))
    have hzODE : ∀ t ∈ Ioo 0 T, HasDerivAt (fun σ => Φ σ 0 v) (A t (Φ t 0 v)) t :=
      fun t ht => by
        have h := (hΦ.2.2 0 t le_rfl ht.1 ht.2).clm_apply (hasDerivAt_const t v)
        simpa using h
    have hdual := ramp_adjoint_duality (1 / lam) T (fun t => Φ t 0 v) p A hδ hδT
      (hcont v) hp hzODE hpd hpT
    have hpne := adjoint_nonzero_from_duality _ _ _ hηne hdual
    obtain ⟨i, hi⟩ := (hε.and (hr.and hG)).exists
    refine plConstants_bddAbove (s i) _ hi.2.2 ?_
    rw [hi.2.1, norm_smul, mul_pow]
    exact mul_pos (pow_pos (norm_pos_iff.2 hi.1) 2) (pow_pos (norm_pos_iff.2 hpne) 2)
  exact ⟨hmain.1, hmain.2, fun μ hμ =>
    (Real.sqrt_le_sqrt (le_optimalPL hbdd hμ)).trans hmain.2⟩

end ModernODE
