import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Data.Real.Sqrt
import Mathlib.Tactic

/-!
# Verified components of the unstable continuous-time PL bound

These lemmas verify the exponential estimate and the Hilbert-space adjoint
ratio in the proof of the manuscript's Theorem 2. They do not construct the
nonlinear forced trajectory or prove its limiting PL ratio.
-/

namespace ModernODE

/-- Endpoint growth and a forward upper bound force growth of the same
perturbation at every intermediate time. This is the scalar estimate in
Step 1 of the unstable proof. -/
theorem intermediate_growth_lower_bound (a b lam T t zT zt : ℝ) (ha : 0 < a)
    (hterminal : b * Real.exp (lam * T) ≤ zT)
    (hpropagate : zT ≤ a * Real.exp (lam * (T - t)) * zt) :
    b / a * Real.exp (lam * t) ≤ zt := by
  have hk : 0 < a * Real.exp (lam * (T - t)) := by positivity
  apply (mul_le_mul_iff_right₀ hk).mp
  calc
    a * Real.exp (lam * (T - t)) * (b / a * Real.exp (lam * t)) =
        b * (Real.exp (lam * (T - t)) * Real.exp (lam * t)) := by
          field_simp [ha.ne']
    _ = b * Real.exp (lam * T) := by
      rw [← Real.exp_add]
      congr 2
      ring
    _ ≤ zT := hterminal
    _ ≤ a * Real.exp (lam * (T - t)) * zt := hpropagate

/-- Algebraic passage from the two integral energy estimates in Step 2 to
the residual-to-trajectory ratio. The integral estimates are explicit inputs. -/
theorem unstable_residual_ratio (a b lam T δ U E : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hlam : 0 < lam) (hδ : 0 < δ) (hδT : δ < T)
    (hU : U ≤ a ^ 2 / (2 * lam * δ ^ 2) * (Real.exp (2 * lam * δ) - 1))
    (hE : b ^ 2 / (2 * lam * a ^ 2) *
      (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ)) ≤ E) :
    U / E ≤ a ^ 4 / (b ^ 2 * δ ^ 2) *
      ((Real.exp (2 * lam * δ) - 1) /
        (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ))) := by
  have hdiff : 0 < Real.exp (2 * lam * T) - Real.exp (2 * lam * δ) := by
    apply sub_pos.mpr
    apply Real.exp_lt_exp.mpr
    nlinarith
  have hexp : 0 ≤ Real.exp (2 * lam * δ) - 1 := by
    apply sub_nonneg.mpr
    apply Real.one_le_exp_iff.mpr
    positivity
  have hlow : 0 < b ^ 2 / (2 * lam * a ^ 2) *
      (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ)) := by positivity
  have hEpos : 0 < E := hlow.trans_le hE
  have hupper : 0 ≤ a ^ 2 / (2 * lam * δ ^ 2) * (Real.exp (2 * lam * δ) - 1) := by
    positivity
  calc
    U / E ≤ (a ^ 2 / (2 * lam * δ ^ 2) * (Real.exp (2 * lam * δ) - 1)) / E :=
      div_le_div_of_nonneg_right hU hEpos.le
    _ ≤ (a ^ 2 / (2 * lam * δ ^ 2) * (Real.exp (2 * lam * δ) - 1)) /
        (b ^ 2 / (2 * lam * a ^ 2) *
          (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ))) :=
      div_le_div_of_nonneg_left hupper hlow hE
    _ = a ^ 4 / (b ^ 2 * δ ^ 2) *
        ((Real.exp (2 * lam * δ) - 1) /
          (Real.exp (2 * lam * T) - Real.exp (2 * lam * δ))) := by
      field_simp [ha.ne', hb.ne', hlam.ne', hδ.ne', hdiff.ne']

/-- The residual ratio after choosing the ramp duration `δ = 1 / lam`.
The two assumptions are precisely the energy estimates at that duration. -/
theorem unstable_energy_ratio_at_inverse_rate (a b lam T U E : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hlam : 0 < lam) (horizon : 2 ≤ lam * T)
    (hU : U ≤ (a ^ 2 * lam / 2) * (Real.exp 2 - 1))
    (hE : b ^ 2 / (2 * lam * a ^ 2) *
      (Real.exp (2 * lam * T) - Real.exp 2) ≤ E) :
    U / E ≤ (a ^ 4 / b ^ 2) * lam ^ 2 *
      ((Real.exp 2 - 1) / (Real.exp (2 * lam * T) - Real.exp 2)) := by
  have hδ : 0 < 1 / lam := by positivity
  have hδT : 1 / lam < T := by
    apply (div_lt_iff₀ hlam).mpr
    nlinarith
  have hcancel : 2 * lam * (1 / lam) = 2 := by field_simp [hlam.ne']
  have hcoeff : a ^ 2 / (2 * lam * (1 / lam) ^ 2) = a ^ 2 * lam / 2 := by
    field_simp [hlam.ne']
  have h := unstable_residual_ratio a b lam T (1 / lam) U E ha hb hlam hδ hδT
    (by simpa only [hcancel, hcoeff] using hU) (by simpa only [hcancel] using hE)
  rw [hcancel] at h
  convert h using 1
  field_simp [hlam.ne']

/-- The last scalar estimate in the proof of Theorem 2. -/
theorem unstable_exponential_ratio (lam T : ℝ) (horizon : 2 ≤ lam * T) :
    (Real.exp 2 - 1) / (Real.exp (2 * lam * T) - Real.exp 2) ≤
      Real.exp 2 * Real.exp (-(2 * lam * T)) := by
  have hX : 0 < Real.exp (2 * lam * T) := Real.exp_pos _
  have hden : 0 < Real.exp (2 * lam * T) - Real.exp 2 := by
    apply sub_pos.mpr
    apply Real.exp_lt_exp.mpr
    linarith
  have hsquare : (Real.exp 2) ^ 2 ≤ Real.exp (2 * lam * T) := by
    rw [sq, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    linarith
  have hratio :
      (Real.exp 2 - 1) / (Real.exp (2 * lam * T) - Real.exp 2) ≤
        Real.exp 2 / Real.exp (2 * lam * T) := by
    apply (div_le_div_iff₀ hden hX).mpr
    nlinarith
  simpa [Real.exp_neg, div_eq_mul_inv] using hratio

/-- The squared constant from Theorem 2, assuming the preceding residual-ratio bound. -/
theorem unstable_squared_constant (a b lam T μ : ℝ) (horizon : 2 ≤ lam * T)
    (hratio : μ ≤ (a ^ 4 / b ^ 2) * lam ^ 2 *
      ((Real.exp 2 - 1) / (Real.exp (2 * lam * T) - Real.exp 2))) :
    μ ≤ (Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T))) ^ 2 := by
  have hfac : 0 ≤ (a ^ 4 / b ^ 2) * lam ^ 2 := by positivity
  have h := mul_le_mul_of_nonneg_left (unstable_exponential_ratio lam T horizon) hfac
  have hexp : Real.exp 2 = (Real.exp 1) ^ 2 := by
    rw [sq, ← Real.exp_add]
    norm_num
  have hexp' : Real.exp (-(2 * lam * T)) = (Real.exp (-(lam * T))) ^ 2 := by
    rw [sq, ← Real.exp_add]
    congr 1
    ring
  calc
    μ ≤ (a ^ 4 / b ^ 2) * lam ^ 2 *
        (Real.exp 2 * Real.exp (-(2 * lam * T))) := hratio.trans h
    _ = (Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T))) ^ 2 := by
      rw [hexp, hexp']
      ring

/-- The displayed square-root constant of Theorem 2, from the residual-ratio bound. -/
theorem unstable_sqrt_constant (a b lam T μ : ℝ) (hb : 0 < b) (hlam : 0 < lam)
    (horizon : 2 ≤ lam * T)
    (hratio : μ ≤ (a ^ 4 / b ^ 2) * lam ^ 2 *
      ((Real.exp 2 - 1) / (Real.exp (2 * lam * T) - Real.exp 2))) :
    Real.sqrt μ ≤ Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)) := by
  have hbound := unstable_squared_constant a b lam T μ horizon hratio
  have hnonneg : 0 ≤ Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)) := by
    positivity
  exact (Real.sqrt_le_iff).mpr ⟨hnonneg, hbound⟩

/-- For fixed parameters the square-root upper bound decays to zero as
the time horizon tends to infinity. -/
theorem unstable_upper_bound_tendsto_zero (a b lam : ℝ) (hlam : 0 < lam) :
    Filter.Tendsto
      (fun T : ℝ => Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T)))
      Filter.atTop (nhds 0) := by
  have hlin : Filter.Tendsto (fun T : ℝ => lam * T) Filter.atTop Filter.atTop :=
    Filter.tendsto_id.const_mul_atTop hlam
  have hexp := Real.tendsto_exp_neg_atTop_nhds_zero.comp hlin
  simpa only [mul_zero] using hexp.const_mul (Real.exp 1 * lam * a ^ 2 / b)

/-- Any nonnegative family of PL constants satisfying the unstable
square-root bound eventually tends to zero. Parameters are fixed while the
time horizon tends to infinity. This is the asymptotic consequence stated
in Theorem 2, separated from supplying the horizon-dependent ODE data. -/
theorem unstable_pl_family_tendsto_zero (a b lam : ℝ) (hlam : 0 < lam)
    (μ : ℝ → ℝ)
    (hμ : ∀ᶠ T in Filter.atTop, 0 ≤ μ T)
    (hbound : ∀ᶠ T in Filter.atTop, Real.sqrt (μ T) ≤
      Real.exp 1 * lam * a ^ 2 / b * Real.exp (-(lam * T))) :
    Filter.Tendsto μ Filter.atTop (nhds 0) := by
  have hsqrt : Filter.Tendsto (fun T => Real.sqrt (μ T)) Filter.atTop (nhds 0) :=
    squeeze_zero' (Filter.Eventually.of_forall (fun T => Real.sqrt_nonneg (μ T)))
      hbound (unstable_upper_bound_tendsto_zero a b lam hlam)
  have hsquare : Filter.Tendsto (fun T => (Real.sqrt (μ T)) ^ 2)
      Filter.atTop (nhds (0 : ℝ)) := by
    simpa only [zero_pow (by norm_num : (2 : ℕ) ≠ 0)] using hsqrt.pow 2
  apply Filter.Tendsto.congr' _ hsquare
  filter_upwards [hμ] with T hT
  exact Real.sq_sqrt hT

/-- A nonzero perturbation in the adjoint duality identity forces a nonzero
adjoint residual, as needed to form its PL ratio. -/
theorem adjoint_nonzero_from_duality {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℝ H] (η p u : H) (hη : η ≠ 0)
    (hdual : ‖η‖ ^ 2 = inner ℝ p u) : p ≠ 0 := by
  intro hp
  have hηpos : 0 < ‖η‖ := norm_pos_iff.mpr hη
  simp only [hp, inner_zero_left] at hdual
  nlinarith

/-- Cauchy--Schwarz converts the adjoint duality identity to the ratio used in
Step 3 of the unstable proof. The identity must be proved by integration by
parts in an ODE instantiation. -/
theorem adjoint_ratio_from_duality {H : Type*} [NormedAddCommGroup H]
    [InnerProductSpace ℝ H] (η p u : H) (hη : η ≠ 0)
    (hdual : ‖η‖ ^ 2 = inner ℝ p u) :
    ‖η‖ ^ 2 / ‖p‖ ^ 2 ≤ ‖u‖ ^ 2 / ‖η‖ ^ 2 := by
  have hηpos : 0 < ‖η‖ := norm_pos_iff.mpr hη
  have hp : p ≠ 0 := adjoint_nonzero_from_duality η p u hη hdual
  have hppos : 0 < ‖p‖ := norm_pos_iff.mpr hp
  have hcs : ‖η‖ ^ 2 ≤ ‖p‖ * ‖u‖ := by
    rw [hdual]
    exact real_inner_le_norm p u
  apply (div_le_div_iff₀ (sq_pos_of_pos hppos) (sq_pos_of_pos hηpos)).mpr
  have hsq := mul_self_le_mul_self (sq_nonneg ‖η‖) hcs
  nlinarith

end ModernODE
