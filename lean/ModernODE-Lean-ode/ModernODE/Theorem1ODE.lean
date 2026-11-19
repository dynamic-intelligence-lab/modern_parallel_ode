import ModernODE.TransitionODE

/-!
# Theorem 1 from the IVP: only the manuscript's hypotheses remain

`theorem1_ode` also took `hfin`: one candidate with nonzero residual and a
finite first-variation bound. It is needed only because `optimalPL` is a real
supremum (for `E = 0` every residual vanishes and the real `sSup` is `0`).
Here it is proved for `E ≠ 0` from the exact solution (`exists_finite_slope`):

* follow the exact solution `x*` up to `T - h`;
* on `[T-h, T]` solve the forced IVP `ṡ = f(s,t) + φ(t)e` (short-time
  Picard–Lindelöf), where `φ` is a smooth bump supported in `(T-h, T)`;
* the residual is `φ e = (∫_t^T q) e` with `q = -φ'`; by Fubini its pairing
  with `v` equals the pairing of `q e` with `∫₀ᵗ v`, so the slope is finite.

`theorem1_ode'` then has exactly the manuscript's hypotheses: `f` C¹ in the
state, an exact solution of the IVP, and the exponential bound on the
state-transition operator of every candidate (`a ≥ 1`, `c > 0`, `T > 0`).
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace ModernODE

open MeasureTheory Set Filter Topology Metric
open scoped NNReal

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-! ### Integration by parts against a tail integral -/

/-- Fubini on the triangle: `∫₀ᵀ (∫_t^T q) g = ∫₀ᵀ q(σ) ∫₀^σ g`. -/
theorem integral_tail_mul_eq {T : ℝ} {q g : ℝ → ℝ} (hq : IntegrableOn q (Ioc 0 T))
    (hg : IntegrableOn g (Ioc 0 T)) :
    ∫ t in Ioc 0 T, (∫ σ in t..T, q σ) * g t =
      ∫ σ in Ioc 0 T, q σ * ∫ t in (0 : ℝ)..σ, g t := by
  set K : ℝ × ℝ → ℝ := {p : ℝ × ℝ | p.2 ≤ p.1}.indicator fun p => q p.1 * g p.2 with hKdef
  have hK : Integrable K
      ((volume.restrict (Ioc (0 : ℝ) T)).prod (volume.restrict (Ioc (0 : ℝ) T))) :=
    (Integrable.mul_prod hq hg).indicator (measurableSet_le measurable_snd measurable_fst)
  have hin1 : ∀ σ ∈ Ioc (0 : ℝ) T,
      ∫ t in Ioc (0 : ℝ) T, K (σ, t) = q σ * ∫ t in (0 : ℝ)..σ, g t := by
    intro σ hσ
    have : (fun t => K (σ, t)) = (Iic σ).indicator fun t => q σ * g t := by
      funext t
      simp only [hKdef, indicator, mem_setOf_eq, mem_Iic]
    rw [this, setIntegral_indicator measurableSet_Iic, Ioc_inter_Iic, inf_eq_right.2 hσ.2,
      integral_const_mul, intervalIntegral.integral_of_le hσ.1.le]
  have hin2 : ∀ t ∈ Ioc (0 : ℝ) T,
      ∫ σ in Ioc (0 : ℝ) T, K (σ, t) = (∫ σ in t..T, q σ) * g t := by
    intro t ht
    have : (fun σ => K (σ, t)) = (Ici t).indicator fun σ => q σ * g t := by
      funext σ
      simp only [hKdef, indicator, mem_setOf_eq, mem_Ici]
    have hset : Ioc (0 : ℝ) T ∩ Ici t = Icc t T := by
      ext x
      simp only [mem_inter_iff, mem_Ioc, mem_Ici, mem_Icc]
      constructor
      · rintro ⟨⟨_, h2⟩, h3⟩
        exact ⟨h3, h2⟩
      · rintro ⟨h1, h2⟩
        exact ⟨⟨ht.1.trans_le h1, h2⟩, h1⟩
    rw [this, setIntegral_indicator measurableSet_Ici, hset, integral_mul_const,
      integral_Icc_eq_integral_Ioc, intervalIntegral.integral_of_le ht.2]
  calc ∫ t in Ioc 0 T, (∫ σ in t..T, q σ) * g t
      = ∫ t in Ioc (0 : ℝ) T, ∫ σ in Ioc (0 : ℝ) T, K (σ, t) :=
        setIntegral_congr_fun measurableSet_Ioc fun t ht => (hin2 t ht).symm
    _ = ∫ σ in Ioc (0 : ℝ) T, ∫ t in Ioc (0 : ℝ) T, K (σ, t) :=
        (integral_integral_swap (f := fun σ t => K (σ, t)) hK).symm
    _ = ∫ σ in Ioc 0 T, q σ * ∫ t in (0 : ℝ)..σ, g t :=
        setIntegral_congr_fun measurableSet_Ioc hin1

/-- A tail-integral residual `(∫_t^T q) e` pairs with `v` only through the
trajectory variation: `⟨(∫_t^T q) e, v⟩ = ⟨q e, ∫₀ᵗ v⟩` in `L²(0,T)`. -/
theorem inner_tail_eq {T : ℝ} {q : ℝ → ℝ} (hq : ContinuousOn q (Icc 0 T)) (e : E)
    (hφ : MemLp (fun t => (∫ σ in t..T, q σ) • e) 2 (volume.restrict (Ioc 0 T)))
    (hQ : MemLp (fun σ => q σ • e) 2 (volume.restrict (Ioc 0 T))) (v : IntervalL2 E T) :
    inner ℝ (hφ.toLp _ : IntervalL2 E T) v =
      inner ℝ (hQ.toLp _ : IntervalL2 E T) (primL2 T v) := by
  rw [MeasureTheory.L2.inner_def, MeasureTheory.L2.inner_def]
  have h1 : (fun t => inner ℝ ((hφ.toLp _ : IntervalL2 E T) t) (v t))
      =ᵐ[volume.restrict (Ioc 0 T)] fun t => (∫ σ in t..T, q σ) * inner ℝ e (v t) := by
    filter_upwards [hφ.coeFn_toLp] with t ht
    rw [ht, real_inner_smul_left]
  have h2 : (fun σ => inner ℝ ((hQ.toLp _ : IntervalL2 E T) σ) ((primL2 T v : IntervalL2 E T) σ))
      =ᵐ[volume.restrict (Ioc 0 T)] fun σ => q σ * ∫ t in (0 : ℝ)..σ, inner ℝ e (v t) := by
    filter_upwards [hQ.coeFn_toLp, coeFn_primL2 v, ae_restrict_mem measurableSet_Ioc]
      with σ hσ1 hσ2 hσ3
    rw [hσ1, hσ2, real_inner_smul_left]
    congr 1
    unfold primFun
    exact ((innerSL ℝ e).intervalIntegral_comp_comm
      (intervalIntegrable_L2 v (Ioc_subset_Icc_self hσ3))).symm
  rw [integral_congr_ae h1, integral_congr_ae h2]
  exact integral_tail_mul_eq (hq.integrableOn_Icc.mono_set Ioc_subset_Icc_self)
    (Integrable.const_inner e (integrableOn_Ioc_L2 v))

/-! ### A finite slope bound for a tail-integral residual -/

theorem norm_jacApplyL2_le (F : C1Field E) (x₀ : E) {T : ℝ} (U W : IntervalL2 E T) {M : ℝ}
    (hM : ∀ t ∈ Icc 0 T, ‖F.J (traj x₀ U t) t‖ ≤ M) :
    ‖jacApplyL2 F x₀ U (continuousOn_primFun W)‖ ≤ M * ‖primL2 T W‖ := by
  unfold jacApplyL2
  apply Lp.norm_le_mul_norm_of_ae_le_mul
  filter_upwards [(memLp_of_continuousOn
      (continuousOn_jacApply F x₀ U (continuousOn_primFun W))).coeFn_toLp,
    coeFn_primL2 W, ae_restrict_mem measurableSet_Ioc] with t h1 h2 h3
  rw [h1, h2]
  exact ((F.J _ t).le_opNorm _).trans
    (mul_le_mul_of_nonneg_right (hM t (Ioc_subset_Icc_self h3)) (norm_nonneg _))

/-- If the residual is `(∫_t^T q) e` with `q` continuous, the first variation
is bounded by a multiple of the trajectory norm `‖∫₀ᵗ v‖`. -/
theorem boundsVariation_of_tail (F : C1Field E) (x₀ : E) {T : ℝ} (u : DerivCoord E T)
    {q : ℝ → ℝ} (hq : ContinuousOn q (Icc 0 T)) (e : E)
    (hφ : MemLp (fun t => (∫ σ in t..T, q σ) • e) 2 (volume.restrict (Ioc 0 T)))
    (hr : residual F x₀ T u = hφ.toLp _) :
    ∃ G, BoundsVariation (meritVariation (residual F x₀ T) u) G := by
  have hQ : MemLp (fun σ => q σ • e) 2 (volume.restrict (Ioc 0 T)) :=
    memLp_of_continuousOn (hq.smul continuousOn_const)
  obtain ⟨M, hM⟩ := isCompact_Icc.exists_bound_of_continuousOn
    (continuousOn_jacAlong F x₀ (DerivCoord.toL2 u))
  refine ⟨‖hQ.toLp _‖ + ‖residual F x₀ T u‖ * max M 0,
    add_nonneg (norm_nonneg _) (mul_nonneg (norm_nonneg _) (le_max_right _ _)), fun v => ?_⟩
  have hv : ‖v‖ = ‖primL2 T (DerivCoord.toL2 v)‖ := DerivCoord.norm_def v
  have hjac := norm_jacApplyL2_le F x₀ (DerivCoord.toL2 u) (DerivCoord.toL2 v)
    (M := max M 0) fun t ht => (hM t ht).trans (le_max_left _ _)
  have h1 : inner ℝ (residual F x₀ T u) (DerivCoord.toL2 v) =
      inner ℝ (hQ.toLp _ : IntervalL2 E T) (primL2 T (DerivCoord.toL2 v)) := by
    rw [hr]
    exact inner_tail_eq hq e hφ hQ _
  rw [meritVariation_eq _ _ _ _ (hasDerivAt_residual F x₀ u v), inner_sub_right, h1, hv]
  set P := primL2 T (DerivCoord.toL2 v)
  set Jv := jacApplyL2 F x₀ (DerivCoord.toL2 u) (continuousOn_primFun (DerivCoord.toL2 v))
  calc |inner ℝ (hQ.toLp _ : IntervalL2 E T) P - inner ℝ (residual F x₀ T u) Jv|
      = |inner ℝ (hQ.toLp _ : IntervalL2 E T) P + -inner ℝ (residual F x₀ T u) Jv| := by
        rw [sub_eq_add_neg]
    _ ≤ |inner ℝ (hQ.toLp _ : IntervalL2 E T) P| + |-inner ℝ (residual F x₀ T u) Jv| :=
        abs_add_le _ _
    _ ≤ ‖hQ.toLp _‖ * ‖P‖ + ‖residual F x₀ T u‖ * (max M 0 * ‖P‖) := by
        rw [abs_neg]
        exact add_le_add (abs_real_inner_le_norm _ _)
          ((abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_left hjac (norm_nonneg _)))
    _ = (‖hQ.toLp _‖ + ‖residual F x₀ T u‖ * max M 0) * ‖P‖ := by ring

/-! ### Short-time forced solution near `t = T` (Picard–Lindelöf) -/

/-- Near the endpoint, the forced IVP `ẇ = f(w,t) + ρ(t)` started from `y(T-h)`
has a solution on `[T-h, T]`, for one `h > 0` that works for every continuous
forcing bounded by `B`. -/
theorem exists_forced_solution (F : C1Field E) {T : ℝ} (hT : 0 < T) {y : ℝ → E}
    (hy : ContinuousOn y (Icc 0 T)) (B : ℝ) :
    ∃ h > 0, h ≤ T ∧ ∀ ρ : ℝ → E, Continuous ρ → (∀ t, ‖ρ t‖ ≤ B) →
      ∃ w : ℝ → E, w (T - h) = y (T - h) ∧
        ∀ t ∈ Icc (T - h) T, HasDerivWithinAt w (F.f (w t) t + ρ t) (Icc (T - h) T) t := by
  -- `J` is bounded by `K` on a neighbourhood of `(y T, T)`.
  obtain ⟨ε₁, hε₁, hJ⟩ := Metric.continuousAt_iff.1
    (F.continuous_J.continuousAt (x := (y T, T))) 1 one_pos
  set ε := ε₁ / 2 with hεdef
  have hε : 0 < ε := by positivity
  set K := ‖F.J (y T) T‖ + 1 with hKdef
  have hK0 : 0 ≤ K := by positivity
  have hJK : ∀ z t, ‖z - y T‖ ≤ ε → |t - T| ≤ ε → ‖F.J z t‖ ≤ K := by
    intro z t hz ht
    have hd : dist (z, t) (y T, T) < ε₁ := by
      rw [Prod.dist_eq, dist_eq_norm, Real.dist_eq]
      exact (max_le hz ht).trans_lt (by linarith)
    have h1 := hJ hd
    simp only [dist_eq_norm] at h1
    have h2 := norm_sub_norm_le (F.J z t) (F.J (y T) T)
    linarith
  have hlip : ∀ t, |t - T| ≤ ε →
      LipschitzOnWith ⟨K, hK0⟩ (fun z => F.f z t) (closedBall (y T) ε) := by
    intro t ht
    refine (convex_closedBall (y T) ε).lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
      (f' := fun z => F.J z t) (fun z _ => (F.hasFDerivAt z t).hasFDerivWithinAt)
      fun z hz => ?_
    rw [← NNReal.coe_le_coe, coe_nnnorm]
    exact hJK z t (mem_closedBall_iff_norm.1 hz) ht
  -- `f(y T, ·)` is bounded near `T`.
  have hfc : Continuous fun t => F.f (y T) t :=
    F.continuous_f.comp (continuous_const.prodMk continuous_id)
  obtain ⟨Cf, hCf⟩ := (isCompact_Icc (a := T - ε) (b := T)).exists_bound_of_continuousOn
    hfc.continuousOn
  set L := max Cf 0 + K * ε + max B 0 + 1 with hLdef
  have hL : 0 < L := by
    have h1 := le_max_right Cf 0
    have h2 := le_max_right B 0
    have h3 := mul_nonneg hK0 hε.le
    linarith
  -- `y` is continuous at `T`.
  obtain ⟨δ, hδ, hyδ⟩ := Metric.continuousWithinAt_iff.1 (hy T ⟨hT.le, le_rfl⟩) (ε / 2)
    (by positivity)
  set h := min (min ε T) (min (δ / 2) (ε / (2 * L))) with hhdef
  have hh : 0 < h := lt_min (lt_min hε hT) (lt_min (by positivity) (by positivity))
  have hhε : h ≤ ε := (min_le_left _ _).trans (min_le_left _ _)
  have hhT : h ≤ T := (min_le_left _ _).trans (min_le_right _ _)
  have hhδ : h ≤ δ / 2 := (min_le_right _ _).trans (min_le_left _ _)
  have hhL : L * h ≤ ε / 2 := by
    have h1 : h ≤ ε / (2 * L) := (min_le_right _ _).trans (min_le_right _ _)
    calc L * h ≤ L * (ε / (2 * L)) := mul_le_mul_of_nonneg_left h1 hL.le
      _ = ε / 2 := by field_simp
  refine ⟨h, hh, hhT, fun ρ hρ hρB => ?_⟩
  have hx : y (T - h) ∈ closedBall (y T) (ε / 2) := by
    rw [mem_closedBall]
    refine (hyδ ⟨by linarith, by linarith⟩ ?_).le
    rw [Real.dist_eq, abs_of_neg (by linarith)]
    linarith
  have hnear : ∀ t ∈ Icc (T - h) T, |t - T| ≤ ε := fun t ht => by
    rw [abs_le]
    constructor <;> linarith [ht.1, ht.2]
  have hPL : IsPicardLindelof (fun t z => F.f z t + ρ t) (tmin := T - h) (tmax := T)
      ⟨T - h, le_rfl, by linarith⟩ (y T) ⟨ε, hε.le⟩ ⟨ε / 2, by positivity⟩ ⟨L, hL.le⟩
      ⟨K, hK0⟩ := by
    refine ⟨fun t ht => ?_, fun z _ => ?_, fun t ht z hz => ?_, ?_⟩
    · have hl := hlip t (hnear t ht)
      exact LipschitzOnWith.of_dist_le_mul fun z1 hz1 z2 hz2 => by
        rw [dist_add_right]
        exact hl.dist_le_mul z1 hz1 z2 hz2
    · exact ((F.continuous_f.comp (continuous_const.prodMk continuous_id)).add hρ).continuousOn
    · have h1 := (hlip t (hnear t ht)).dist_le_mul z hz (y T) (mem_closedBall_self hε.le)
      rw [dist_eq_norm, dist_eq_norm] at h1
      have h2 : ‖F.f (y T) t‖ ≤ Cf := hCf t ⟨by linarith [ht.1, hhε], ht.2⟩
      have h3 : ‖z - y T‖ ≤ ε := mem_closedBall_iff_norm.1 hz
      have h4 := norm_add_le (F.f z t) (ρ t)
      have h5 := norm_sub_norm_le (F.f z t) (F.f (y T) t)
      have h6 := mul_le_mul_of_nonneg_left h3 hK0
      have h7 := hρB t
      have h8 := le_max_left Cf 0
      have h9 := le_max_left B 0
      show ‖F.f z t + ρ t‖ ≤ L
      simp only [NNReal.coe_mk] at h1
      rw [hLdef]
      nlinarith
    · show L * max (T - (T - h)) ((T - h) - (T - h)) ≤ ε - ε / 2
      rw [sub_sub_cancel, sub_self, max_eq_left hh.le]
      linarith
  obtain ⟨w, hw0, hw⟩ := hPL.exists_eq_forall_mem_Icc_hasDerivWithinAt hx
  exact ⟨w, hw0, hw⟩

/-! ### Gluing the forced solution to the exact solution -/

theorem intervalIntegrable_L2_sub {T : ℝ} (u : IntervalL2 E T) {a b : ℝ} (ha : 0 ≤ a)
    (hab : a ≤ b) (hb : b ≤ T) : IntervalIntegrable (u : ℝ → E) volume a b :=
  (intervalIntegrable_iff_integrableOn_Ioc_of_le hab).2
    ((integrableOn_Ioc_L2 u).mono_set (Ioc_subset_Ioc ha hb))

/-- Follow the exact solution on `[0, τ₀]` and a forced solution `w` on
`[τ₀, T]`: the glued candidate has residual exactly the forcing `ρ`. -/
theorem exists_glued_candidate (F : C1Field E) (x₀ : E) {T τ₀ : ℝ} (hτ₀ : 0 ≤ τ₀)
    (hτ₀T : τ₀ ≤ T) {u : DerivCoord E T} (hu : residual F x₀ T u = 0)
    {ρ : ℝ → E} (hρ : Continuous ρ) (hρ0 : ∀ t ≤ τ₀, ρ t = 0)
    {w : ℝ → E} (hw0 : w τ₀ = traj x₀ (DerivCoord.toL2 u) τ₀)
    (hw : ∀ t ∈ Icc τ₀ T, HasDerivWithinAt w (F.f (w t) t + ρ t) (Icc τ₀ T) t) :
    ∃ uC : DerivCoord E T,
      residual F x₀ T uC = (memLp_of_continuousOn hρ.continuousOn).toLp ρ := by
  set U := DerivCoord.toL2 u with hUdef
  -- The exact solution: `U = f(x*, ·)` a.e.
  have hUae : (U : ℝ → E) =ᵐ[volume.restrict (Ioc 0 T)] fieldAlong F x₀ U := by
    have h' : U = (memLp_of_continuousOn (continuousOn_fieldAlong F x₀ U)).toLp _ :=
      sub_eq_zero.1 hu
    have h2 := (memLp_of_continuousOn (continuousOn_fieldAlong F x₀ U)).coeFn_toLp
    rw [← h'] at h2
    exact h2
  have hwc : ContinuousOn w (Icc τ₀ T) := fun t ht => (hw t ht).continuousWithinAt
  -- The forced velocity, extended continuously to all of `ℝ`.
  set G : ℝ → E := fun t => F.f (w (projIcc τ₀ T hτ₀T t)) t + ρ t with hGdef
  have hGc : Continuous G :=
    (F.continuous_f.comp ((hwc.comp_continuous (continuous_subtype_val.comp continuous_projIcc)
      fun t => (projIcc τ₀ T hτ₀T t).2).prodMk continuous_id)).add hρ
  have hGw : ∀ t ∈ Icc τ₀ T, G t = F.f (w t) t + ρ t := fun t ht => by
    simp only [hGdef, projIcc_of_mem hτ₀T ht]
  set ũ : ℝ → E := (Iic τ₀).indicator (U : ℝ → E) + (Ioi τ₀).indicator G with hũdef
  have hmem : MemLp ũ 2 (volume.restrict (Ioc 0 T)) :=
    ((Lp.memLp U).indicator measurableSet_Iic).add
      ((memLp_of_continuousOn hGc.continuousOn).indicator measurableSet_Ioi)
  set uC : DerivCoord E T := hmem.toLp ũ with huCdef
  have hcoe : ∀ᵐ σ ∂volume, σ ∈ Ioc 0 T → (DerivCoord.toL2 uC : ℝ → E) σ = ũ σ :=
    (ae_restrict_iff' measurableSet_Ioc).1 hmem.coeFn_toLp
  -- The trajectory of the glued candidate.
  have hprim1 : ∀ t ∈ Icc 0 τ₀, primFun (DerivCoord.toL2 uC) t = primFun U t := by
    intro t ht
    rw [primFun_congr (u := DerivCoord.toL2 uC) hmem.coeFn_toLp ⟨ht.1, ht.2.trans hτ₀T⟩]
    unfold primFun
    refine intervalIntegral.integral_congr fun σ hσ => ?_
    rw [uIcc_of_le ht.1] at hσ
    have h1 : σ ≤ τ₀ := hσ.2.trans ht.2
    have h2 : ¬ τ₀ < σ := not_lt.2 h1
    simp only [hũdef, Pi.add_apply, indicator_of_mem (show σ ∈ Iic τ₀ from h1),
      indicator_of_notMem (show σ ∉ Ioi τ₀ from h2), add_zero]
  have hprim2 : ∀ t ∈ Icc τ₀ T,
      primFun (DerivCoord.toL2 uC) t = primFun U τ₀ + (w t - w τ₀) := by
    intro t ht
    have hsplit := intervalIntegral.integral_add_adjacent_intervals
      (intervalIntegrable_L2_sub (DerivCoord.toL2 uC) le_rfl hτ₀ hτ₀T)
      (intervalIntegrable_L2_sub (DerivCoord.toL2 uC) hτ₀ ht.1 ht.2)
    have hτ₀mem : τ₀ ∈ Icc 0 τ₀ := ⟨hτ₀, le_rfl⟩
    have hA : ∫ σ in (0 : ℝ)..τ₀, (DerivCoord.toL2 uC : ℝ → E) σ = primFun U τ₀ :=
      hprim1 τ₀ hτ₀mem
    have hB : ∫ σ in τ₀..t, (DerivCoord.toL2 uC : ℝ → E) σ = ∫ σ in τ₀..t, G σ := by
      refine intervalIntegral.integral_congr_ae ?_
      filter_upwards [hcoe] with σ hσ hmem'
      rw [uIoc_of_le ht.1] at hmem'
      rw [hσ ⟨hτ₀.trans_lt hmem'.1, hmem'.2.trans ht.2⟩]
      have h1 : ¬ σ ≤ τ₀ := not_le.2 hmem'.1
      simp only [hũdef, Pi.add_apply, indicator_of_notMem (show σ ∉ Iic τ₀ from h1),
        indicator_of_mem (show σ ∈ Ioi τ₀ from hmem'.1), zero_add]
    have hC : ∫ σ in τ₀..t, G σ = w t - w τ₀ := by
      refine intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le ht.1
        (hwc.mono (Icc_subset_Icc le_rfl ht.2)) (fun x hx => ?_) (hGc.intervalIntegrable _ _)
      have hxT : x ∈ Icc τ₀ T := ⟨hx.1.le, hx.2.le.trans ht.2⟩
      rw [hGw x hxT]
      exact (hw x hxT).hasDerivAt (Icc_mem_nhds hx.1 (hx.2.trans_le ht.2))
    show ∫ σ in (0 : ℝ)..t, (DerivCoord.toL2 uC : ℝ → E) σ = _
    rw [← hsplit, hA, hB, hC]
  refine ⟨uC, ?_⟩
  unfold residual
  show hmem.toLp ũ - _ = _
  rw [← MemLp.toLp_sub]
  apply MemLp.toLp_congr
  have hUae' : ∀ᵐ σ ∂volume, σ ∈ Ioc 0 T → (U : ℝ → E) σ = fieldAlong F x₀ U σ :=
    (ae_restrict_iff' measurableSet_Ioc).1 hUae
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  filter_upwards [hUae'] with σ hσU hσ
  simp only [Pi.sub_apply, hũdef, Pi.add_apply]
  rcases le_or_gt σ τ₀ with h1 | h1
  · have h2 : ¬ τ₀ < σ := not_lt.2 h1
    rw [indicator_of_mem (show σ ∈ Iic τ₀ from h1), indicator_of_notMem (show σ ∉ Ioi τ₀ from h2),
      add_zero,
      hσU hσ, hρ0 σ h1]
    simp only [fieldAlong, traj, hprim1 σ ⟨hσ.1.le, h1⟩, sub_self]
  · have h2 : ¬ σ ≤ τ₀ := not_le.2 h1
    rw [indicator_of_notMem (show σ ∉ Iic τ₀ from h2), indicator_of_mem (show σ ∈ Ioi τ₀ from h1),
      zero_add,
      hGw σ ⟨h1.le, hσ.2⟩]
    simp only [fieldAlong, traj, hprim2 σ ⟨h1.le, hσ.2⟩]
    have hs : x₀ + (primFun U τ₀ + (w σ - w τ₀)) = w σ := by
      rw [hw0]
      simp only [traj]
      abel
    rw [hs, add_sub_cancel_left]

/-! ### The finite-slope candidate -/

/-- For `E ≠ 0` and an exact solution, some candidate has a nonzero residual
and a finite first-variation bound (the hypothesis `hfin` of `theorem1_ode`). -/
theorem exists_finite_slope [Nontrivial E] (F : C1Field E) (x₀ : E) {T : ℝ} (hT : 0 < T)
    (hsol : ∃ u, residual F x₀ T u = 0) :
    ∃ u G, BoundsVariation (meritVariation (residual F x₀ T) u) G ∧
      0 < ‖residual F x₀ T u‖ ^ 2 := by
  obtain ⟨u, hu⟩ := hsol
  obtain ⟨e, he⟩ := exists_ne (0 : E)
  obtain ⟨h, hh, hhT, hsolve⟩ :=
    exists_forced_solution F hT (continuousOn_traj x₀ (DerivCoord.toL2 u)) ‖e‖
  -- A smooth bump `φ` supported in `(T-h, T)`, equal to `1` near `T - h/2`.
  let φ : ContDiffBump (T - h / 2) := ⟨h / 4, h / 2, by positivity, by linarith⟩
  have hφc : ContDiff ℝ 1 φ := φ.contDiff
  have hφ0 : ∀ t, (t ≤ T - h ∨ T ≤ t) → φ t = 0 := fun t ht => φ.zero_of_le_dist (by
    show h / 2 ≤ dist t (T - h / 2)
    rw [Real.dist_eq]
    rcases ht with ht | ht
    · rw [abs_of_nonpos (by linarith)]
      linarith
    · rw [abs_of_nonneg (by linarith)]
      linarith)
  have hρc : Continuous fun t => φ t • e := hφc.continuous.smul continuous_const
  have hρB : ∀ t, ‖φ t • e‖ ≤ ‖e‖ := fun t => by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg φ.nonneg]
    exact mul_le_of_le_one_left (norm_nonneg _) φ.le_one
  obtain ⟨w, hw0, hw⟩ := hsolve (fun t => φ t • e) hρc hρB
  obtain ⟨uC, huC⟩ := exists_glued_candidate F x₀ (by linarith : (0 : ℝ) ≤ T - h)
    (by linarith) hu hρc (fun t ht => by simp only [hφ0 t (Or.inl ht), zero_smul]) hw0 hw
  -- The residual `φ e` is the tail integral of `q = -φ'`.
  have htail : ∀ t, (∫ σ in t..T, -deriv φ σ) = φ t := fun t => by
    rw [intervalIntegral.integral_neg, intervalIntegral.integral_deriv_eq_sub
      (fun x _ => (hφc.differentiable le_rfl) x)
      ((hφc.continuous_deriv le_rfl).intervalIntegrable _ _), hφ0 T (Or.inr le_rfl)]
    ring
  have hφmem : MemLp (fun t => (∫ σ in t..T, -deriv φ σ) • e) 2
      (volume.restrict (Ioc 0 T)) :=
    memLp_of_continuousOn (hρc.continuousOn.congr fun t _ => by simp only [htail])
  have hr : residual F x₀ T uC = hφmem.toLp _ := by
    rw [huC]
    apply MemLp.toLp_congr
    exact Eventually.of_forall fun t => by simp only [htail]
  obtain ⟨G, hG⟩ := boundsVariation_of_tail F x₀ uC (q := fun σ => -deriv φ σ)
    (hφc.continuous_deriv le_rfl).neg.continuousOn e hφmem hr
  refine ⟨uC, G, hG, ?_⟩
  -- The residual is nonzero: `φ e = e ≠ 0` on `(T - 3h/4, T - h/4)`.
  have hne : (memLp_of_continuousOn (T := T) hρc.continuousOn).toLp (fun t => φ t • e) ≠ 0 := by
    intro h0
    have hae : (fun t => φ t • e) =ᵐ[volume.restrict (Ioc (0 : ℝ) T)] 0 := by
      have h1 := (memLp_of_continuousOn (T := T) hρc.continuousOn).coeFn_toLp
      rw [h0] at h1
      exact h1.symm.trans (Lp.coeFn_zero ..)
    rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc, ae_iff] at hae
    have hsub : Ioo (T - 3 * h / 4) (T - h / 4) ⊆
        {t | ¬(t ∈ Ioc 0 T → (fun t => φ t • e) t = (0 : ℝ → E) t)} := by
      intro t ht
      simp only [mem_setOf_eq, _root_.not_imp, Pi.zero_apply]
      refine ⟨⟨by linarith [ht.1], by linarith [ht.2]⟩, ?_⟩
      rw [φ.one_of_mem_closedBall, one_smul]
      · exact he
      · show dist t (T - h / 2) ≤ h / 4
        rw [Real.dist_eq, abs_le]
        constructor <;> linarith [ht.1, ht.2]
    have hnull := measure_mono_null hsub hae
    rw [Real.volume_Ioo, ENNReal.ofReal_eq_zero] at hnull
    linarith
  rw [huC]
  exact pow_pos (norm_pos_iff.2 hne) 2

/-! ### Theorem 1 with the manuscript's hypotheses only -/

/-- **Theorem 1 from the IVP.** Hypotheses, as in the manuscript: `f` is C¹ in
the state with continuous `J = ∂ₓf` (`C1Field`), the IVP has an exact solution
(`hsol`), `a ≥ 1`, `c > 0`, `T > 0`, and the state-transition operator of the
linearization along every candidate satisfies `‖Φ(t,τ)‖ ≤ a e^{-c(t-τ)}`.
By `exists_isTransition` and `isTransition_unique` there is exactly one `Φ`
satisfying `IsTransition`, so `hdecay` is a condition on that operator.
`E ≠ 0` replaces the manuscript's `ℝ^D`, `D ≥ 1`. -/
theorem theorem1_ode' [Nontrivial E] (F : C1Field E) (x₀ : E) {T a c : ℝ} (ha : 1 ≤ a)
    (hc : 0 < c) (hT : 0 < T)
    (hdecay : ∀ (u : DerivCoord E T) Φ,
      IsTransition (fun t => F.J (traj x₀ (DerivCoord.toL2 u) t) t) T Φ →
      ∀ τ t, 0 ≤ τ → τ ≤ t → t ≤ T → ‖Φ t τ‖ ≤ a * Real.exp (-c * (t - τ)))
    (hsol : ∃ u, residual F x₀ T u = 0) :
    sInf (Set.range (fun u => residualLoss (residual F x₀ T u))) = 0 ∧
    IsPL (residual F x₀ T) (meritVariation (residual F x₀ T))
      (((c / a) / (1 - Real.exp (-c * T))) ^ 2) ∧
    (c / a) / (1 - Real.exp (-c * T)) ≤
      Real.sqrt (optimalPL (residual F x₀ T) (meritVariation (residual F x₀ T))) ∧
    c / a ≤ (c / a) / (1 - Real.exp (-c * T)) ∧ 0 < c / a := by
  choose Φ hΦ using fun u : DerivCoord E T =>
    exists_isTransition hT.le (continuousOn_jacAlong F x₀ (DerivCoord.toL2 u))
  exact theorem1_ode F x₀ Φ (fun u => (hΦ u).1) (fun u => (hΦ u).2.1) (fun u => (hΦ u).2.2)
    ha hc hT (fun u => hdecay u (Φ u) (hΦ u)) hsol (exists_finite_slope F x₀ hT hsol)

end ModernODE
