import ModernODE.Theorem1
import ModernODE.IntervalL2

/-!
# Theorem 1 from the ODE itself

This file removes the correction hypothesis of `StableAnalyticData`
(Steps 1 and 2 of the appendix proof) for a concrete residual map.

* Candidates are `s(t) = x₀ + ∫₀ᵗ u` with `u ∈ L²(0,T)`; in one time
  dimension this is exactly `{s ∈ H¹ : s(0) = x₀}`.
* The variation norm is the L² norm of the trajectory variation `∫₀ᵗ v`.
* The residual is the actual `L²` element `u − f(s(·),·)`.
* `Φ` is passed in through its defining equations: `Φ(τ,τ) = I`,
  `∂ₜΦ(t,τ) = J(s(t),t) Φ(t,τ)`, continuity on the time triangle.
  Existence and uniqueness of such `Φ` are proved in `TransitionODE`.
-/

noncomputable section

set_option linter.unusedSectionVars false

namespace ModernODE

open MeasureTheory Set Filter Topology

/-- A vector field that is C¹ in the state, with jointly continuous partial
derivative `J = ∂ₓ f`. -/
structure C1Field (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E] where
  f : E → ℝ → E
  J : E → ℝ → E →L[ℝ] E
  continuous_f : Continuous fun p : E × ℝ => f p.1 p.2
  continuous_J : Continuous fun p : E × ℝ => J p.1 p.2
  hasFDerivAt : ∀ x t, HasFDerivAt (fun y => f y t) (J x t) x

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

section Prim

variable {T : ℝ}

/-- The primitive `t ↦ ∫₀ᵗ u`. -/
def primFun (u : IntervalL2 E T) (t : ℝ) : E := ∫ τ in (0 : ℝ)..t, (u : ℝ → E) τ

theorem integrableOn_Ioc_L2 (u : IntervalL2 E T) : IntegrableOn (u : ℝ → E) (Ioc 0 T) :=
  MemLp.integrable one_le_two (Lp.memLp u)

theorem integrableOn_Icc_L2 (u : IntervalL2 E T) : IntegrableOn (u : ℝ → E) (Icc 0 T) :=
  (integrableOn_Icc_iff_integrableOn_Ioc (f := (u : ℝ → E)) (μ := volume) (a := 0) (b := T)).2
    (integrableOn_Ioc_L2 u)

theorem intervalIntegrable_L2 (u : IntervalL2 E T) {t : ℝ} (ht : t ∈ Icc 0 T) :
    IntervalIntegrable (u : ℝ → E) volume 0 t :=
  (intervalIntegrable_iff_integrableOn_Ioc_of_le ht.1).2
    ((integrableOn_Ioc_L2 u).mono_set (Ioc_subset_Ioc_right ht.2))

theorem continuousOn_primFun (u : IntervalL2 E T) : ContinuousOn (primFun u) (Icc 0 T) := by
  rcases le_or_gt 0 T with hT | hT
  · have h := intervalIntegral.continuousOn_primitive_interval (μ := volume) (a := 0) (b := T)
      (f := (u : ℝ → E)) (by rw [uIcc_of_le hT]; exact integrableOn_Icc_L2 u)
    rw [uIcc_of_le hT] at h
    exact h
  · rw [Icc_eq_empty (not_le.2 hT)]
    exact continuousOn_empty _

theorem memLp_of_continuousOn {g : ℝ → E} (hg : ContinuousOn g (Icc 0 T)) :
    MemLp g 2 (volume.restrict (Ioc 0 T)) := by
  rcases le_or_gt 0 T with hT | hT
  · exact continuousOn_memLp_two hT hg
  · rw [Ioc_eq_empty (not_lt.2 hT.le), Measure.restrict_empty]
    exact memLp_measure_zero

/-- Two a.e.-equal integrands have the same primitive on `[0,T]`. -/
theorem primFun_congr {u : IntervalL2 E T} {g : ℝ → E}
    (h : (u : ℝ → E) =ᵐ[volume.restrict (Ioc 0 T)] g) {t : ℝ} (ht : t ∈ Icc 0 T) :
    primFun u t = ∫ τ in (0 : ℝ)..t, g τ := by
  unfold primFun
  apply intervalIntegral.integral_congr_ae
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc] at h
  filter_upwards [h] with τ hτ hmem
  rw [uIoc_of_le ht.1] at hmem
  exact hτ ⟨hmem.1, hmem.2.trans ht.2⟩

theorem primFun_add (u w : IntervalL2 E T) {t : ℝ} (ht : t ∈ Icc 0 T) :
    primFun (u + w) t = primFun u t + primFun w t := by
  rw [primFun_congr (Lp.coeFn_add u w) ht]
  exact intervalIntegral.integral_add (intervalIntegrable_L2 u ht) (intervalIntegrable_L2 w ht)

theorem primFun_smul (c : ℝ) (u : IntervalL2 E T) {t : ℝ} (ht : t ∈ Icc 0 T) :
    primFun (c • u) t = c • primFun u t := by
  rw [primFun_congr (Lp.coeFn_smul c u) ht]
  exact intervalIntegral.integral_smul c _

/-- The primitive as a linear map on `L²`. -/
def primL2 (T : ℝ) : IntervalL2 E T →ₗ[ℝ] IntervalL2 E T where
  toFun u := (memLp_of_continuousOn (continuousOn_primFun u)).toLp (primFun u)
  map_add' u w := by
    rw [← MemLp.toLp_add]
    apply MemLp.toLp_congr
    rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
    exact Eventually.of_forall fun t ht => primFun_add u w (Ioc_subset_Icc_self ht)
  map_smul' c u := by
    rw [RingHom.id_apply, ← MemLp.toLp_const_smul]
    apply MemLp.toLp_congr
    rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
    exact Eventually.of_forall fun t ht => primFun_smul c u (Ioc_subset_Icc_self ht)

theorem coeFn_primL2 (u : IntervalL2 E T) :
    (primL2 T u : ℝ → E) =ᵐ[volume.restrict (Ioc 0 T)] primFun u :=
  (memLp_of_continuousOn (continuousOn_primFun u)).coeFn_toLp

end Prim

/-! ## Derivative coordinates with the trajectory L² norm -/

/-- `u` stands for the trajectory `x₀ + ∫₀ᵗ u`. The norm is the L² norm of
`∫₀ᵗ u`, the trajectory norm used by the manuscript. -/
def DerivCoord (E : Type*) [NormedAddCommGroup E] [InnerProductSpace ℝ E] (T : ℝ) : Type _ :=
  IntervalL2 E T

namespace DerivCoord

variable {T : ℝ}

instance : AddCommGroup (DerivCoord E T) := inferInstanceAs (AddCommGroup (IntervalL2 E T))

instance : Module ℝ (DerivCoord E T) := inferInstanceAs (Module ℝ (IntervalL2 E T))

/-- The underlying `L²` time derivative. -/
def toL2 : DerivCoord E T →ₗ[ℝ] IntervalL2 E T :=
  { toFun := fun u => u, map_add' := fun _ _ => rfl, map_smul' := fun _ _ => rfl }

/-- The trajectory variation `∫₀ᵗ u`. -/
def trajL2 : DerivCoord E T →ₗ[ℝ] IntervalL2 E T := (primL2 T).comp toL2

instance : SeminormedAddCommGroup (DerivCoord E T) :=
  SeminormedAddCommGroup.induced _ _ (trajL2 (E := E) (T := T))

instance : NormedSpace ℝ (DerivCoord E T) :=
  NormedSpace.induced ℝ _ _ (trajL2 (E := E) (T := T))

theorem norm_def (u : DerivCoord E T) : ‖u‖ = ‖trajL2 u‖ := rfl

/-- Every `L²` function is a derivative coordinate. -/
def ofL2 : IntervalL2 E T →ₗ[ℝ] DerivCoord E T :=
  { toFun := fun u => u, map_add' := fun _ _ => rfl, map_smul' := fun _ _ => rfl }

@[simp] theorem toL2_ofL2 (u : IntervalL2 E T) : toL2 (ofL2 u) = u := rfl

end DerivCoord

/-! ## The concrete residual map -/

section Residual

variable (F : C1Field E) (x₀ : E) {T : ℝ}

/-- The trajectory `x₀ + ∫₀ᵗ u`. -/
def traj (u : IntervalL2 E T) (t : ℝ) : E := x₀ + primFun u t

theorem continuousOn_traj (u : IntervalL2 E T) : ContinuousOn (traj x₀ u) (Icc 0 T) :=
  continuousOn_const.add (continuousOn_primFun u)

/-- The vector field along a trajectory. -/
def fieldAlong (u : IntervalL2 E T) (t : ℝ) : E := F.f (traj x₀ u t) t

theorem continuousOn_fieldAlong (u : IntervalL2 E T) :
    ContinuousOn (fieldAlong F x₀ u) (Icc 0 T) :=
  F.continuous_f.comp_continuousOn ((continuousOn_traj x₀ u).prodMk continuousOn_id)

/-- The Jacobian along a trajectory. -/
theorem continuousOn_jacAlong (u : IntervalL2 E T) :
    ContinuousOn (fun t => F.J (traj x₀ u t) t) (Icc 0 T) :=
  F.continuous_J.comp_continuousOn ((continuousOn_traj x₀ u).prodMk continuousOn_id)

/-- The actual residual `ṡ − f(s,t)` of the candidate with derivative `u`. -/
def residual (T : ℝ) (u : DerivCoord E T) : IntervalL2 E T :=
  DerivCoord.toL2 u -
    (memLp_of_continuousOn (continuousOn_fieldAlong F x₀ (DerivCoord.toL2 u))).toLp _

end Residual

/-! ## Step 1: the first variation of the residual -/

section Step1

variable (F : C1Field E)

/-- Uniform first-order expansion of the vector field along a continuous path.
This is where continuity of `J = ∂ₓ f` enters. -/
theorem nemytskii_pointwise {T : ℝ} {s η : ℝ → E}
    (hs : ContinuousOn s (Icc 0 T)) (hη : ContinuousOn η (Icc 0 T)) {δ : ℝ} (hδ : 0 < δ) :
    ∃ ρ > 0, ∀ h : ℝ, |h| < ρ → ∀ t ∈ Icc 0 T,
      ‖F.f (s t + h • η t) t - F.f (s t) t - h • F.J (s t) t (η t)‖ ≤ δ * (|h| * ‖η t‖) := by
  set K : Set (ℝ × ℝ) := Icc 0 T ×ˢ Icc (-1) 1
  have hK : IsCompact K := isCompact_Icc.prod isCompact_Icc
  have hcont : ContinuousOn (fun p : ℝ × ℝ => F.J (s p.1 + p.2 • η p.1) p.1) K := by
    have h1 : ContinuousOn (fun p : ℝ × ℝ => s p.1 + p.2 • η p.1) K :=
      (hs.comp continuousOn_fst (fun p hp => hp.1)).add
        (continuousOn_snd.smul (hη.comp continuousOn_fst (fun p hp => hp.1)))
    exact F.continuous_J.comp_continuousOn (h1.prodMk continuousOn_fst)
  obtain ⟨ρ, hρ, hρK⟩ :=
    Metric.uniformContinuousOn_iff.1 (hK.uniformContinuousOn_of_continuous hcont) δ hδ
  refine ⟨min ρ 1, lt_min hρ one_pos, fun h hh t ht => ?_⟩
  have hh1 : |h| < 1 := hh.trans_le (min_le_right _ _)
  have hhρ : |h| < ρ := hh.trans_le (min_le_left _ _)
  have hderiv : ∀ y ∈ segment ℝ (s t) (s t + h • η t),
      HasFDerivWithinAt (fun y => F.f y t) (F.J y t) (segment ℝ (s t) (s t + h • η t)) y :=
    fun y _ => (F.hasFDerivAt y t).hasFDerivWithinAt
  have hbound : ∀ y ∈ segment ℝ (s t) (s t + h • η t), ‖F.J y t - F.J (s t) t‖ ≤ δ := by
    intro y hy
    rw [segment_eq_image'] at hy
    obtain ⟨θ, hθ, rfl⟩ := hy
    have hθh : |θ * h| ≤ |h| := by
      rw [abs_mul, abs_of_nonneg hθ.1]
      exact mul_le_of_le_one_left (abs_nonneg h) hθ.2
    have hmem1 : (t, θ * h) ∈ K := ⟨ht, abs_le.1 (hθh.trans hh1.le)⟩
    have hmem0 : (t, (0 : ℝ)) ∈ K := ⟨ht, by norm_num⟩
    have hdist : dist (t, θ * h) (t, (0 : ℝ)) < ρ := by
      rw [Prod.dist_eq, dist_self, Real.dist_eq, sub_zero]
      exact (max_le (abs_nonneg _) le_rfl).trans_lt (hθh.trans_lt hhρ)
    have key := hρK (t, θ * h) hmem1 (t, 0) hmem0 hdist
    rw [dist_eq_norm] at key
    simp only [zero_smul, add_zero] at key
    simp only [add_sub_cancel_left, smul_smul]
    exact key.le
  have hmv := (convex_segment (s t) (s t + h • η t)).norm_image_sub_le_of_norm_hasFDerivWithin_le'
    hderiv hbound (left_mem_segment ℝ _ _) (right_mem_segment ℝ _ _)
  rw [add_sub_cancel_left, map_smul, norm_smul, Real.norm_eq_abs] at hmv
  exact hmv

variable (x₀ : E) {T : ℝ}

theorem continuousOn_jacApply (u : IntervalL2 E T) {η : ℝ → E} (hη : ContinuousOn η (Icc 0 T)) :
    ContinuousOn (fun t => F.J (traj x₀ u t) t (η t)) (Icc 0 T) :=
  (continuousOn_jacAlong F x₀ u).clm_apply hη

/-- The linearized term `J(s(t),t) η(t)` as an `L²` element. -/
def jacApplyL2 (u : IntervalL2 E T) {η : ℝ → E} (hη : ContinuousOn η (Icc 0 T)) :
    IntervalL2 E T :=
  (memLp_of_continuousOn (continuousOn_jacApply F x₀ u hη)).toLp _

theorem traj_add_smul (U W : IntervalL2 E T) (h : ℝ) {t : ℝ} (ht : t ∈ Icc 0 T) :
    traj x₀ (U + h • W) t = traj x₀ U t + h • primFun W t := by
  unfold traj
  rw [primFun_add U (h • W) ht, primFun_smul h W ht, add_assoc]

/-- **Step 1.** The residual has the first variation
`Dr[u](v) = v − J(s(·),·) ∫₀ᵗ v` in `L²`, derived from `f ∈ C¹`. -/
theorem hasDerivAt_residual (u v : DerivCoord E T) :
    HasDerivAt (fun ε : ℝ => residual F x₀ T (u + ε • v))
      (DerivCoord.toL2 v - jacApplyL2 F x₀ (DerivCoord.toL2 u)
        (continuousOn_primFun (DerivCoord.toL2 v))) 0 := by
  set U := DerivCoord.toL2 u
  set W := DerivCoord.toL2 v
  have hη : ContinuousOn (primFun W) (Icc 0 T) := continuousOn_primFun W
  obtain ⟨M, hM⟩ := isCompact_Icc.exists_bound_of_continuousOn hη
  set M' := max M 0
  set C : ℝ := (measureUnivNNReal (volume.restrict (Ioc (0 : ℝ) T)) : ℝ) ^
    ((2 : ENNReal).toReal⁻¹)
  have hC : 0 ≤ C := by positivity
  rw [hasDerivAt_iff_isLittleO_nhds_zero]
  refine Asymptotics.IsLittleO.of_bound fun c hc => ?_
  have hpos : 0 < C * M' + 1 := by positivity
  obtain ⟨ρ, hρ, hbound⟩ :=
    nemytskii_pointwise F (continuousOn_traj x₀ U) hη (δ := c / (C * M' + 1)) (by positivity)
  filter_upwards [Metric.ball_mem_nhds (0 : ℝ) hρ] with h hh
  have hh' : |h| < ρ := by simpa [Real.dist_eq] using hh
  set hN1 := memLp_of_continuousOn (continuousOn_fieldAlong F x₀ (U + h • W))
  set hN0 := memLp_of_continuousOn (continuousOn_fieldAlong F x₀ U)
  set Jη := jacApplyL2 F x₀ U hη
  set G : IntervalL2 E T := hN1.toLp _ - hN0.toLp _ - h • Jη
  have hU : DerivCoord.toL2 (u + h • v) = U + h • W := by
    rw [map_add, map_smul]
  have hrem : residual F x₀ T (u + (0 + h) • v) - residual F x₀ T (u + (0 : ℝ) • v) -
      h • (W - Jη) = -G := by
    simp only [zero_add, zero_smul, add_zero, residual, hU, G]
    module
  rw [hrem, norm_neg, Real.norm_eq_abs]
  have hGae : ∀ᵐ t ∂(volume.restrict (Ioc (0 : ℝ) T)),
      ‖G t‖ ≤ c / (C * M' + 1) * (|h| * M') := by
    filter_upwards [Lp.coeFn_sub (hN1.toLp _ - hN0.toLp _) (h • Jη),
      Lp.coeFn_sub (hN1.toLp _) (hN0.toLp _), Lp.coeFn_smul h Jη, hN1.coeFn_toLp,
      hN0.coeFn_toLp,
      (memLp_of_continuousOn (continuousOn_jacApply F x₀ U hη)).coeFn_toLp,
      ae_restrict_mem measurableSet_Ioc] with t h1 h2 h3 h4 h5 h6 ht
    have htI : t ∈ Icc 0 T := Ioc_subset_Icc_self ht
    have h6' : (Jη : ℝ → E) t = F.J (traj x₀ U t) t (primFun W t) := h6
    simp only [G]
    rw [h1, Pi.sub_apply, h2, Pi.sub_apply, h3, Pi.smul_apply, h4, h5, h6']
    simp only [fieldAlong]
    rw [traj_add_smul x₀ U W h htI]
    refine (hbound h hh' t htI).trans ?_
    gcongr
    exact (hM t htI).trans (le_max_left _ _)
  have hGnorm := Lp.norm_le_of_ae_bound (by positivity) hGae
  calc ‖G‖ ≤ C * (c / (C * M' + 1) * (|h| * M')) := hGnorm
    _ = (C * M' / (C * M' + 1)) * c * |h| := by field_simp
    _ ≤ 1 * c * |h| := by
        gcongr
        exact (div_le_one hpos).2 (by linarith)
    _ = c * |h| := by ring

/-- The actual merit is differentiable along every line. -/
theorem merit_differentiable_residual (u v : DerivCoord E T) :
    DifferentiableAt ℝ (fun ε : ℝ => residualLoss (residual F x₀ T (u + ε • v))) 0 :=
  (hasDerivAt_residualLoss (hasDerivAt_residual F x₀ u v)).differentiableAt

end Step1

/-! ## Step 2: variation of constants, by Fubini and the fundamental theorem -/

section Step2

/-- The time triangle `0 ≤ τ ≤ t ≤ T`, as pairs `(t, τ)`. -/
def timeTriangle (T : ℝ) : Set (ℝ × ℝ) := {p | 0 ≤ p.2 ∧ p.2 ≤ p.1 ∧ p.1 ≤ T}

theorem isClosed_timeTriangle (T : ℝ) : IsClosed (timeTriangle T) :=
  (isClosed_le continuous_const continuous_snd).inter
    ((isClosed_le continuous_snd continuous_fst).inter (isClosed_le continuous_fst continuous_const))

theorem isCompact_timeTriangle (T : ℝ) : IsCompact (timeTriangle T) :=
  (isCompact_Icc.prod (isCompact_Icc (a := (0 : ℝ)) (b := T))).of_isClosed_subset
    (isClosed_timeTriangle T) fun _ hp => ⟨⟨hp.1.trans hp.2.1, hp.2.2⟩, ⟨hp.1, hp.2.1.trans hp.2.2⟩⟩

theorem measurableSet_timeTriangle (T : ℝ) : MeasurableSet (timeTriangle T) :=
  (isClosed_timeTriangle T).measurableSet

/-- An operator family continuous on `[a,b]`, applied to an integrable path. -/
theorem integrableOn_clm_apply {Ψ : ℝ → E →L[ℝ] E} {R : ℝ → E} {a b : ℝ}
    (hΨ : ContinuousOn Ψ (Icc a b)) (hR : IntegrableOn R (Ioc a b)) :
    IntegrableOn (fun τ => Ψ τ (R τ)) (Ioc a b) := by
  obtain ⟨C, hC⟩ := isCompact_Icc.exists_bound_of_continuousOn hΨ
  have hmeas : AEStronglyMeasurable (fun τ => Ψ τ (R τ)) (volume.restrict (Ioc a b)) := by
    simpa using (ContinuousLinearMap.id ℝ (E →L[ℝ] E)).aestronglyMeasurable_comp₂
      ((hΨ.aestronglyMeasurable measurableSet_Icc).mono_measure
        (Measure.restrict_mono Ioc_subset_Icc_self le_rfl)) hR.aestronglyMeasurable
  refine Integrable.mono' (hR.norm.const_mul (max C 0)) hmeas ?_
  rw [ae_restrict_iff' measurableSet_Ioc]
  refine Eventually.of_forall fun τ hτ => ?_
  calc ‖Ψ τ (R τ)‖ ≤ ‖Ψ τ‖ * ‖R τ‖ := (Ψ τ).le_opNorm _
    _ ≤ max C 0 * ‖R τ‖ := by
        gcongr
        exact (hC τ (Ioc_subset_Icc_self hτ)).trans (le_max_left _ _)

variable {T : ℝ} {R : ℝ → E} {A : ℝ → E →L[ℝ] E} {Φ : ℝ → ℝ → E →L[ℝ] E}

/-- The kernel `A(σ) Φ(σ,τ)` of the variation-of-constants identity. -/
def vocKernel (A : ℝ → E →L[ℝ] E) (Φ : ℝ → ℝ → E →L[ℝ] E) (p : ℝ × ℝ) : E →L[ℝ] E :=
  (A p.1).comp (Φ p.1 p.2)

/-- The integrand on `[0,t]²`, zero off the time triangle. -/
def vocIntegrand (A : ℝ → E →L[ℝ] E) (Φ : ℝ → ℝ → E →L[ℝ] E) (T : ℝ) (R : ℝ → E)
    (p : ℝ × ℝ) : E :=
  (timeTriangle T).indicator (vocKernel A Φ) p (R p.2)

/-- The variation-of-constants path `η(t) = ∫₀ᵗ Φ(t,τ) R(τ) dτ`. -/
def vocPath (Φ : ℝ → ℝ → E →L[ℝ] E) (R : ℝ → E) (t : ℝ) : E :=
  ∫ τ in (0 : ℝ)..t, Φ t τ (R τ)

variable (hA : ContinuousOn A (Icc 0 T))
  (hΦ : ContinuousOn (fun p : ℝ × ℝ => Φ p.1 p.2) (timeTriangle T))
  (hR : IntegrableOn R (Ioc 0 T))
include hA hΦ hR

theorem continuousOn_vocKernel : ContinuousOn (vocKernel A Φ) (timeTriangle T) :=
  (hA.comp continuousOn_fst fun _ hp => ⟨hp.1.trans hp.2.1, hp.2.2⟩).clm_comp hΦ

theorem integrable_vocIntegrand {t : ℝ} (ht : t ∈ Icc 0 T) :
    Integrable (vocIntegrand A Φ T R)
      ((volume.restrict (Ioc 0 t)).prod (volume.restrict (Ioc 0 t))) := by
  have hK := continuousOn_vocKernel hA hΦ hR
  obtain ⟨C, hC⟩ := (isCompact_timeTriangle T).exists_bound_of_continuousOn hK
  have hRt : Integrable R (volume.restrict (Ioc 0 t)) := hR.mono_set (Ioc_subset_Ioc_right ht.2)
  have hmeasK : AEStronglyMeasurable ((timeTriangle T).indicator (vocKernel A Φ))
      ((volume.restrict (Ioc 0 t)).prod (volume.restrict (Ioc 0 t))) :=
    (aestronglyMeasurable_indicator_iff (measurableSet_timeTriangle T)).2
      (hK.aestronglyMeasurable (measurableSet_timeTriangle T))
  have hmeas : AEStronglyMeasurable (vocIntegrand A Φ T R)
      ((volume.restrict (Ioc 0 t)).prod (volume.restrict (Ioc 0 t))) := by
    simpa [vocIntegrand] using (ContinuousLinearMap.id ℝ (E →L[ℝ] E)).aestronglyMeasurable_comp₂
      hmeasK hRt.aestronglyMeasurable.comp_snd
  refine Integrable.mono' ((integrable_const (max C 0)).mul_prod hRt.norm) hmeas
    (Eventually.of_forall fun p => ?_)
  calc ‖vocIntegrand A Φ T R p‖
      ≤ ‖(timeTriangle T).indicator (vocKernel A Φ) p‖ * ‖R p.2‖ :=
        ContinuousLinearMap.le_opNorm _ _
    _ ≤ max C 0 * ‖R p.2‖ := by
        gcongr
        by_cases hp : p ∈ timeTriangle T
        · rw [indicator_of_mem hp]
          exact (hC p hp).trans (le_max_left _ _)
        · rw [indicator_of_notMem hp, norm_zero]
          exact le_max_right _ _

theorem vocIntegrand_inner_tau {t σ : ℝ} (ht : t ∈ Icc 0 T) (hσ : σ ∈ Ioc 0 t) :
    ∫ τ, vocIntegrand A Φ T R (σ, τ) ∂(volume.restrict (Ioc 0 t)) =
      A σ (vocPath Φ R σ) := by
  have hσT : σ ≤ T := hσ.2.trans ht.2
  have heq : EqOn (fun τ => vocIntegrand A Φ T R (σ, τ))
      ((Iic σ).indicator fun τ => A σ (Φ σ τ (R τ))) (Ioc 0 t) := by
    intro τ hτ
    by_cases hτσ : τ ≤ σ
    · have hmem : (σ, τ) ∈ timeTriangle T := ⟨hτ.1.le, hτσ, hσT⟩
      simp [vocIntegrand, indicator_of_mem hmem, indicator_of_mem (show τ ∈ Iic σ from hτσ),
        vocKernel]
    · have hmem : (σ, τ) ∉ timeTriangle T := fun h => hτσ h.2.1
      simp [vocIntegrand, indicator_of_notMem hmem,
        indicator_of_notMem (show τ ∉ Iic σ from hτσ)]
  have hΦσ : ContinuousOn (fun τ => Φ σ τ) (Icc 0 σ) :=
    hΦ.comp (continuousOn_const.prodMk continuousOn_id) fun τ hτ => ⟨hτ.1, hτ.2, hσT⟩
  rw [setIntegral_congr_fun measurableSet_Ioc heq, setIntegral_indicator measurableSet_Iic,
    Ioc_inter_Iic, min_eq_right hσ.2, ← intervalIntegral.integral_of_le hσ.1.le]
  exact (A σ).intervalIntegral_comp_comm ((intervalIntegrable_iff_integrableOn_Ioc_of_le hσ.1.le).2
    (integrableOn_clm_apply hΦσ (hR.mono_set (Ioc_subset_Ioc_right hσT))))

theorem integrableOn_voc_marginal (hT : 0 ≤ T) :
    IntegrableOn (fun σ => A σ (vocPath Φ R σ)) (Ioc 0 T) := by
  have h := (integrable_vocIntegrand hA hΦ hR ⟨hT, le_rfl⟩).integral_prod_left
  refine h.congr ?_
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  exact Eventually.of_forall fun σ hσ => vocIntegrand_inner_tau hA hΦ hR ⟨hT, le_rfl⟩ hσ

variable (hΦid : ∀ τ ∈ Icc 0 T, Φ τ τ = 1)
  (hΦd : ∀ τ t, 0 ≤ τ → τ < t → t < T →
    HasDerivAt (fun σ => Φ σ τ) ((A t).comp (Φ t τ)) t)
include hΦid hΦd

theorem vocIntegrand_inner_sigma {t τ : ℝ} (ht : t ∈ Icc 0 T) (hτ : τ ∈ Ioc 0 t) :
    ∫ σ, vocIntegrand A Φ T R (σ, τ) ∂(volume.restrict (Ioc 0 t)) = Φ t τ (R τ) - R τ := by
  have hK := continuousOn_vocKernel hA hΦ hR
  have hmaps : MapsTo (fun σ => (σ, τ)) (Icc τ t) (timeTriangle T) :=
    fun σ hσ => ⟨hτ.1.le, hσ.1, hσ.2.trans ht.2⟩
  have hslice : ContinuousOn (fun σ => vocKernel A Φ (σ, τ)) (Icc τ t) :=
    hK.comp (continuousOn_id.prodMk continuousOn_const) hmaps
  have heq : EqOn (fun σ => vocIntegrand A Φ T R (σ, τ))
      ((Ici τ).indicator fun σ => vocKernel A Φ (σ, τ) (R τ)) (Ioc 0 t) := by
    intro σ hσ
    by_cases hτσ : τ ≤ σ
    · have hmem : (σ, τ) ∈ timeTriangle T := ⟨hτ.1.le, hτσ, hσ.2.trans ht.2⟩
      simp [vocIntegrand, indicator_of_mem hmem, indicator_of_mem (show σ ∈ Ici τ from hτσ)]
    · have hmem : (σ, τ) ∉ timeTriangle T := fun h => hτσ h.2.1
      simp [vocIntegrand, indicator_of_notMem hmem,
        indicator_of_notMem (show σ ∉ Ici τ from hτσ)]
  have hset : Ioc 0 t ∩ Ici τ = Icc τ t := by
    ext σ
    simp only [mem_inter_iff, mem_Ioc, mem_Ici, mem_Icc]
    constructor
    · rintro ⟨⟨_, h2⟩, h3⟩
      exact ⟨h3, h2⟩
    · rintro ⟨h1, h2⟩
      exact ⟨⟨hτ.1.trans_le h1, h2⟩, h1⟩
  have hint : IntegrableOn (fun σ => vocKernel A Φ (σ, τ)) (Ioc τ t) :=
    hslice.integrableOn_Icc.mono_set Ioc_subset_Icc_self
  have hderiv : ∀ σ ∈ Ioo τ t, HasDerivAt (fun σ => Φ σ τ) (vocKernel A Φ (σ, τ)) σ :=
    fun σ hσ => hΦd τ σ hτ.1.le hσ.1 (hσ.2.trans_le ht.2)
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le hτ.2
    (hΦ.comp (continuousOn_id.prodMk continuousOn_const) hmaps) hderiv
    (hslice.intervalIntegrable_of_Icc hτ.2)
  rw [setIntegral_congr_fun measurableSet_Ioc heq, setIntegral_indicator measurableSet_Ici, hset,
    integral_Icc_eq_integral_Ioc, ← ContinuousLinearMap.integral_apply hint,
    ← intervalIntegral.integral_of_le hτ.2, hftc]
  simp only [Function.comp_apply, id]
  rw [hΦid τ ⟨hτ.1.le, hτ.2.trans ht.2⟩]
  simp

/-- **Step 2.** `η(t) = ∫₀ᵗ Φ(t,τ)R(τ)dτ` satisfies `∫₀ᵗ A η = η(t) − ∫₀ᵗ R`,
the integral form of `η' = Aη + R`, `η(0) = 0`. -/
theorem voc_identity {t : ℝ} (ht : t ∈ Icc 0 T) :
    (∫ σ in (0 : ℝ)..t, A σ (vocPath Φ R σ)) = vocPath Φ R t - ∫ τ in (0 : ℝ)..t, R τ := by
  have hint := integrable_vocIntegrand hA hΦ hR ht
  have hswap := integral_integral_swap (f := fun σ τ => vocIntegrand A Φ T R (σ, τ)) hint
  have hRt : IntegrableOn R (Ioc 0 t) := hR.mono_set (Ioc_subset_Ioc_right ht.2)
  have hΦt : ContinuousOn (fun τ => Φ t τ) (Icc 0 t) :=
    hΦ.comp (continuousOn_const.prodMk continuousOn_id) fun τ hτ => ⟨hτ.1, hτ.2, ht.2⟩
  unfold vocPath
  rw [intervalIntegral.integral_of_le ht.1, intervalIntegral.integral_of_le ht.1,
    intervalIntegral.integral_of_le ht.1, ← integral_sub (integrableOn_clm_apply hΦt hRt) hRt]
  calc ∫ σ in Ioc 0 t, A σ (∫ τ in (0 : ℝ)..σ, Φ σ τ (R τ))
      = ∫ σ in Ioc 0 t, ∫ τ in Ioc 0 t, vocIntegrand A Φ T R (σ, τ) :=
        setIntegral_congr_fun measurableSet_Ioc fun σ hσ =>
          (vocIntegrand_inner_tau hA hΦ hR ht hσ).symm
    _ = ∫ τ in Ioc 0 t, ∫ σ in Ioc 0 t, vocIntegrand A Φ T R (σ, τ) := hswap
    _ = ∫ τ in Ioc 0 t, (Φ t τ (R τ) - R τ) :=
        setIntegral_congr_fun measurableSet_Ioc fun τ hτ =>
          vocIntegrand_inner_sigma hA hΦ hR hΦid hΦd ht hτ

/-- The variation-of-constants path is continuous: it equals
`∫₀ᵗ R + ∫₀ᵗ A η`, a sum of primitives of integrable functions. -/
theorem continuousOn_vocPath : ContinuousOn (vocPath Φ R) (Icc 0 T) := by
  rcases le_or_gt 0 T with hT | hT
  · have hprim : ∀ {g : ℝ → E}, IntegrableOn g (Ioc 0 T) →
        ContinuousOn (fun t => ∫ τ in (0 : ℝ)..t, g τ) (Icc 0 T) := by
      intro g hg
      have h := intervalIntegral.continuousOn_primitive_interval (μ := volume) (a := 0) (b := T)
        (f := g) (by
          rw [uIcc_of_le hT]
          exact (integrableOn_Icc_iff_integrableOn_Ioc (f := g) (μ := volume) (a := 0)
            (b := T)).2 hg)
      rwa [uIcc_of_le hT] at h
    refine ((hprim hR).add (hprim (integrableOn_voc_marginal hA hΦ hR hT))).congr
      fun t ht => ?_
    have h := voc_identity hA hΦ hR hΦid hΦd ht
    show vocPath Φ R t = (∫ τ in (0 : ℝ)..t, R τ) + ∫ σ in (0 : ℝ)..t, A σ (vocPath Φ R σ)
    rw [h]
    abel
  · rw [Icc_eq_empty (not_le.2 hT)]
    exact continuousOn_empty _

end Step2

/-! ## Integrability facts for the stable analytic interface -/

section Interface

variable {T : ℝ}

theorem indicator_norm_sq_eq (φ : ℝ → E) :
    (fun t => ‖(Icc 0 T).indicator φ t‖ ^ 2) = (Icc 0 T).indicator (fun t => ‖φ t‖ ^ 2) := by
  funext t
  by_cases ht : t ∈ Icc 0 T <;> simp [ht]

theorem norm_sq_eq_integral_indicator {g : IntervalL2 E T} {φ : ℝ → E}
    (hgφ : (g : ℝ → E) =ᵐ[volume.restrict (Ioc 0 T)] φ) :
    ‖g‖ ^ 2 = ∫ t, ‖(Icc 0 T).indicator φ t‖ ^ 2 := by
  have h1 : ‖g‖ ^ 2 = ∫ t in Ioc 0 T, ‖φ t‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, L2.inner_def]
    apply integral_congr_ae
    filter_upwards [hgφ] with t ht
    rw [ht, real_inner_self_eq_norm_sq]
  rw [h1, indicator_norm_sq_eq, integral_indicator measurableSet_Icc, integral_Icc_eq_integral_Ioc]

theorem integrable_indicator_sq {φ : ℝ → E} (hφ : MemLp φ 2 (volume.restrict (Ioc 0 T))) :
    Integrable (fun t => ‖(Icc 0 T).indicator φ t‖ ^ 2) := by
  rw [indicator_norm_sq_eq, integrable_indicator_iff measurableSet_Icc]
  exact (integrableOn_Icc_iff_integrableOn_Ioc (f := fun t => ‖φ t‖ ^ 2) (μ := volume) (a := 0)
    (b := T)).2 ((memLp_two_iff_integrable_sq_norm hφ.1).1 hφ)

theorem integrable_indicator_norm {φ : ℝ → E} (hφ : MemLp φ 2 (volume.restrict (Ioc 0 T))) :
    Integrable (fun t => ‖(Icc 0 T).indicator φ t‖) := by
  have h : (fun t => ‖(Icc 0 T).indicator φ t‖) = (Icc 0 T).indicator (fun t => ‖φ t‖) := by
    funext t
    by_cases ht : t ∈ Icc 0 T <;> simp [ht]
  rw [h, integrable_indicator_iff measurableSet_Icc]
  exact (integrableOn_Icc_iff_integrableOn_Ioc (f := fun t => ‖φ t‖) (μ := volume) (a := 0)
    (b := T)).2 (MemLp.integrable one_le_two hφ).norm

theorem volterraKernel_norm_le {a c t τ : ℝ} (ha : 0 ≤ a) (hc : 0 ≤ c) :
    ‖volterraKernel a c T t τ‖ ≤ a := by
  unfold volterraKernel
  split_ifs with h
  · rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    have hexp : Real.exp (-c * (t - τ)) ≤ 1 :=
      Real.exp_le_one_iff.2 (by nlinarith [mul_nonneg hc (sub_nonneg.2 h.2.1)])
    exact mul_le_of_le_one_right ha hexp
  · simpa using ha

theorem volterraKernel_norm_le_indicator {a c t τ : ℝ} (ha : 0 ≤ a) (hc : 0 ≤ c) :
    ‖volterraKernel a c T t τ‖ ≤ (Icc 0 T).indicator (fun _ => a) t := by
  by_cases ht : t ∈ Icc 0 T
  · rw [indicator_of_mem ht]
    exact volterraKernel_norm_le ha hc
  · rw [indicator_of_notMem ht, volterraKernel_row_zero ht]
    simp

theorem measurable_volterraKernel_uncurry (a c T : ℝ) :
    Measurable (fun p : ℝ × ℝ => volterraKernel a c T p.1 p.2) := by
  unfold volterraKernel
  exact Measurable.ite (measurableSet_timeTriangle T) (by fun_prop) measurable_const

theorem integrable_kernel_mul {a c t : ℝ} (ha : 0 ≤ a) (hc : 0 ≤ c) {g : ℝ → ℝ}
    (hg : Integrable g) : Integrable (fun τ => volterraKernel a c T t τ * g τ) :=
  hg.bdd_mul (volterraKernel_integrable a c T t).aestronglyMeasurable
    ⟨a, fun _ => volterraKernel_norm_le ha hc⟩

theorem integrable_kernel_joint {a c : ℝ} (ha : 0 ≤ a) (hc : 0 ≤ c) {g : ℝ → ℝ}
    (hg : Integrable g) :
    Integrable (fun p : ℝ × ℝ => volterraKernel a c T p.1 p.2 * g p.2) (volume.prod volume) := by
  have hind : Integrable ((Icc 0 T).indicator fun _ : ℝ => a) :=
    (continuous_const : Continuous fun _ : ℝ => a).integrableOn_Icc.integrable_indicator
      measurableSet_Icc
  refine Integrable.mono' (hind.mul_prod hg.norm)
    ((measurable_volterraKernel_uncurry a c T).aestronglyMeasurable.mul
      hg.aestronglyMeasurable.comp_snd) (Eventually.of_forall fun p => ?_)
  rw [norm_mul]
  gcongr
  exact volterraKernel_norm_le_indicator ha hc

theorem integral_transitionAction_eq (Ψ : ℝ → ℝ → E →L[ℝ] E) (R : ℝ → E) (t : ℝ) :
    ∫ τ, transitionAction Ψ T ((Icc 0 T).indicator R) t τ =
      (Icc 0 T).indicator (vocPath Ψ R) t := by
  by_cases ht : t ∈ Icc 0 T
  · rw [indicator_of_mem ht]
    have hfun : ∀ τ, transitionAction Ψ T ((Icc 0 T).indicator R) t τ =
        (Icc 0 t).indicator (fun τ => Ψ t τ (R τ)) τ := by
      intro τ
      unfold transitionAction
      by_cases hτ : τ ∈ Icc 0 t
      · have hτT : τ ∈ Icc 0 T := ⟨hτ.1, hτ.2.trans ht.2⟩
        rw [if_pos ⟨hτ.1, hτ.2, ht.2⟩, indicator_of_mem hτ, indicator_of_mem hτT]
      · rw [if_neg (fun h => hτ ⟨h.1, h.2.1⟩), indicator_of_notMem hτ]
    rw [integral_congr_ae (Eventually.of_forall hfun), integral_indicator measurableSet_Icc,
      integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le ht.1]
    rfl
  · rw [indicator_of_notMem ht]
    have hfun : ∀ τ, transitionAction Ψ T ((Icc 0 T).indicator R) t τ = 0 := by
      intro τ
      unfold transitionAction
      rw [if_neg (fun h => ht ⟨h.1.trans h.2.1, h.2.2⟩)]
    rw [integral_congr_ae (Eventually.of_forall hfun)]
    simp

end Interface

/-! ## The correction direction and the conditional-free Theorem 1 -/

section Correction

variable (F : C1Field E) (x₀ : E) {T : ℝ} (Φ : DerivCoord E T → ℝ → ℝ → E →L[ℝ] E)
  (hΦcont : ∀ u, ContinuousOn (fun p : ℝ × ℝ => Φ u p.1 p.2) (timeTriangle T))
  (hΦid : ∀ u τ, τ ∈ Icc 0 T → Φ u τ τ = 1)
  (hΦode : ∀ u τ t, 0 ≤ τ → τ < t → t < T →
    HasDerivAt (fun σ => Φ u σ τ) ((F.J (traj x₀ (DerivCoord.toL2 u) t) t).comp (Φ u t τ)) t)
include hΦcont hΦid hΦode

theorem continuousOn_corrPath (u : DerivCoord E T) :
    ContinuousOn (vocPath (Φ u) (residual F x₀ T u : ℝ → E)) (Icc 0 T) :=
  continuousOn_vocPath (continuousOn_jacAlong F x₀ (DerivCoord.toL2 u)) (hΦcont u)
    (integrableOn_Ioc_L2 _) (hΦid u) (hΦode u)

/-- The correction direction, in derivative coordinates: `η' = r + J(s)η`
for `η = ∫₀ᵗ Φ(t,τ) r(τ) dτ`. -/
def corrDir (u : DerivCoord E T) : DerivCoord E T :=
  DerivCoord.ofL2 (residual F x₀ T u +
    jacApplyL2 F x₀ (DerivCoord.toL2 u) (continuousOn_corrPath F x₀ Φ hΦcont hΦid hΦode u))

/-- The trajectory of the correction is the variation-of-constants path. -/
theorem primFun_corrDir (u : DerivCoord E T) {t : ℝ} (ht : t ∈ Icc 0 T) :
    primFun (DerivCoord.toL2 (corrDir F x₀ Φ hΦcont hΦid hΦode u)) t =
      vocPath (Φ u) (residual F x₀ T u : ℝ → E) t := by
  set U := DerivCoord.toL2 u
  set hη := continuousOn_corrPath F x₀ Φ hΦcont hΦid hΦode u
  have h1 : DerivCoord.toL2 (corrDir F x₀ Φ hΦcont hΦid hΦode u) =
      residual F x₀ T u + jacApplyL2 F x₀ U hη := rfl
  have h2 : primFun (jacApplyL2 F x₀ U hη) t =
      ∫ σ in (0 : ℝ)..t, F.J (traj x₀ U σ) σ (vocPath (Φ u) (residual F x₀ T u : ℝ → E) σ) :=
    primFun_congr (memLp_of_continuousOn (continuousOn_jacApply F x₀ U hη)).coeFn_toLp ht
  rw [h1, primFun_add _ _ ht, h2, voc_identity (continuousOn_jacAlong F x₀ U) (hΦcont u)
    (integrableOn_Ioc_L2 _) (hΦid u) (hΦode u) ht]
  unfold primFun
  abel

/-- **Steps 1 + 2.** The correction's first residual variation is the residual. -/
theorem hasDerivAt_corrDir (u : DerivCoord E T) :
    HasDerivAt (fun ε : ℝ => residual F x₀ T (u + ε • corrDir F x₀ Φ hΦcont hΦid hΦode u))
      (residual F x₀ T u) 0 := by
  refine (hasDerivAt_residual F x₀ u _).congr_deriv ?_
  set U := DerivCoord.toL2 u
  set hη := continuousOn_corrPath F x₀ Φ hΦcont hΦid hΦode u
  have hJ : jacApplyL2 F x₀ U
      (continuousOn_primFun (DerivCoord.toL2 (corrDir F x₀ Φ hΦcont hΦid hΦode u))) =
      jacApplyL2 F x₀ U hη := by
    unfold jacApplyL2
    apply MemLp.toLp_congr
    rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
    exact Eventually.of_forall fun t ht => congrArg (F.J (traj x₀ U t) t)
      (primFun_corrDir F x₀ Φ hΦcont hΦid hΦode u (Ioc_subset_Icc_self ht))
  rw [hJ]
  show residual F x₀ T u + jacApplyL2 F x₀ U hη - jacApplyL2 F x₀ U hη = residual F x₀ T u
  abel

theorem corrPath_eq (u : DerivCoord E T) :
    (fun t => ∫ τ, transitionAction (Φ u) T ((Icc 0 T).indicator (residual F x₀ T u : ℝ → E))
      t τ) = (Icc 0 T).indicator (vocPath (Φ u) (residual F x₀ T u : ℝ → E)) :=
  funext fun t => integral_transitionAction_eq (Φ u) _ t

theorem norm_corrDir_sq (u : DerivCoord E T) :
    ‖corrDir F x₀ Φ hΦcont hΦid hΦode u‖ ^ 2 =
      ∫ t, ‖(Icc 0 T).indicator (vocPath (Φ u) (residual F x₀ T u : ℝ → E)) t‖ ^ 2 := by
  rw [DerivCoord.norm_def]
  apply norm_sq_eq_integral_indicator
  refine (coeFn_primL2 _).trans ?_
  rw [EventuallyEq, ae_restrict_iff' measurableSet_Ioc]
  exact Eventually.of_forall fun t ht =>
    primFun_corrDir F x₀ Φ hΦcont hΦid hΦode u (Ioc_subset_Icc_self ht)

/-- **Theorem 1 from the ODE.** Besides the manuscript's exponential bound on
the state-transition operator, the hypotheses are: `f` continuous with
continuous `∂ₓf` (`C1Field`), `Φ` given by its defining equations, an exact
solution, and one nonzero-residual candidate with finite slope (needed only
because `optimalPL` is a real supremum). -/
theorem theorem1_ode {a c : ℝ} (ha : 1 ≤ a) (hc : 0 < c) (hT : 0 < T)
    (hdecay : ∀ u τ t, 0 ≤ τ → τ ≤ t → t ≤ T → ‖Φ u t τ‖ ≤ a * Real.exp (-c * (t - τ)))
    (hsol : ∃ u, residual F x₀ T u = 0)
    (hfin : ∃ u G, BoundsVariation (meritVariation (residual F x₀ T) u) G ∧
      0 < ‖residual F x₀ T u‖ ^ 2) :
    sInf (Set.range (fun u => residualLoss (residual F x₀ T u))) = 0 ∧
    IsPL (residual F x₀ T) (meritVariation (residual F x₀ T))
      (((c / a) / (1 - Real.exp (-c * T))) ^ 2) ∧
    (c / a) / (1 - Real.exp (-c * T)) ≤
      Real.sqrt (optimalPL (residual F x₀ T) (meritVariation (residual F x₀ T))) ∧
    c / a ≤ (c / a) / (1 - Real.exp (-c * T)) ∧ 0 < c / a := by
  have ha0 : 0 ≤ a := by linarith
  let A : StableAnalyticData (E := E) (residual F x₀ T) a c T :=
    { merit_differentiable := merit_differentiable_residual F x₀
      residualPath := fun u => (Icc 0 T).indicator (residual F x₀ T u : ℝ → E)
      residual_support := fun _ _ ht => indicator_of_notMem ht _
      transition := Φ
      residual_energy := fun _ => norm_sq_eq_integral_indicator EventuallyEq.rfl
      residual_sq_integrable := fun _ => integrable_indicator_sq (Lp.memLp _)
      kernel_residual_integrable := fun _ _ =>
        integrable_kernel_mul ha0 hc.le (integrable_indicator_norm (Lp.memLp _))
      kernel_residual_sq_integrable := fun _ _ =>
        integrable_kernel_mul ha0 hc.le (integrable_indicator_sq (Lp.memLp _))
      joint_integrable := fun _ =>
        integrable_kernel_joint ha0 hc.le (integrable_indicator_sq (Lp.memLp _))
      correction := fun u => ⟨corrDir F x₀ Φ hΦcont hΦid hΦode u,
        fun t => ∫ τ, transitionAction (Φ u) T
          ((Icc 0 T).indicator (residual F x₀ T u : ℝ → E)) t τ,
        hasDerivAt_corrDir F x₀ Φ hΦcont hΦid hΦode u,
        by rw [corrPath_eq F x₀ Φ hΦcont hΦid hΦode u, norm_corrDir_sq F x₀ Φ hΦcont hΦid hΦode u],
        fun _ => rfl,
        by
          rw [corrPath_eq F x₀ Φ hΦcont hΦid hΦode u]
          exact integrable_indicator_sq
            (memLp_of_continuousOn (continuousOn_corrPath F x₀ Φ hΦcont hΦid hΦode u))⟩
      zero_residual := hsol
      finite_nonzero_point := hfin }
  exact theorem1 (residual F x₀ T) a c T ha hc hT A hdecay

end Correction

end ModernODE
