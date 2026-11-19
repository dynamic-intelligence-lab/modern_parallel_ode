import ModernODE.Core
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Topology.Order.Basic

/-!
# Passing PL bounds to a limiting family

These results certify the order and limit argument in Step 3 of Theorem 2.
In an ODE application, construction of the forced trajectory and convergence
of its scaled gradient remain explicit analytic hypotheses.
-/

namespace ModernODE

open Filter
open scoped Topology

variable {S V H ι : Type*} [SeminormedAddCommGroup V] [SeminormedAddCommGroup H]

/-- A family of finite directional-norm bounds with converging PL ratios
bounds the optimal PL constant, including the extended-norm definition. -/
theorem optimalPL_le_of_ratio_tendsto {r : S → H} {d : S → V → ℝ}
    {l : Filter ι} [l.NeBot] (s : ι → S) (G : ι → ℝ) (C : ℝ)
    (hG : ∀ᶠ i in l, BoundsVariation (d (s i)) (G i))
    (hr : ∀ᶠ i in l, 0 < ‖r (s i)‖ ^ 2)
    (hlim : Tendsto (fun i => G i ^ 2 / ‖r (s i)‖ ^ 2) l (𝓝 C)) :
    optimalPL r d ≤ C := by
  apply optimalPL_le
  intro μ hμ
  apply ge_of_tendsto hlim
  filter_upwards [hG, hr] with i hiG hir
  exact isPL_le_ratio hμ (s i) (G i) hiG hir

/-- A scaled gradient and exact scaled residual give the limiting upper bound
used in the nonlinear part of the unstable proof. No differentiability or
ODE-existence claim is hidden in this lemma: the bound on the first variation
and convergence of `g` are explicit hypotheses. -/
theorem optimalPL_le_of_scaled_family
    [NormedSpace ℝ H] {r : S → H} {d : S → V → ℝ}
    {l : Filter ι} [l.NeBot] (s : ι → S) (ε : ι → ℝ) (g : ι → H) (p η : H)
    (hp : ‖p‖ ≠ 0)
    (hε : ∀ᶠ i in l, ε i ≠ 0)
    (hr : ∀ᶠ i in l, r (s i) = ε i • p)
    (hG : ∀ᶠ i in l, BoundsVariation (d (s i)) ‖ε i • g i‖)
    (hlim : Tendsto g l (𝓝 η)) :
    optimalPL r d ≤ ‖η‖ ^ 2 / ‖p‖ ^ 2 := by
  have hratio : ∀ᶠ i in l,
      ‖ε i • g i‖ ^ 2 / ‖r (s i)‖ ^ 2 = ‖g i‖ ^ 2 / ‖p‖ ^ 2 := by
    filter_upwards [hε, hr] with i hiε hir
    rw [hir, norm_smul, norm_smul, mul_pow, mul_pow]
    exact mul_div_mul_left _ _ (pow_ne_zero _ (norm_ne_zero_iff.mpr hiε))
  apply optimalPL_le_of_ratio_tendsto s (fun i => ‖ε i • g i‖) _ hG
  · filter_upwards [hε, hr] with i hiε hir
    rw [hir, norm_smul, mul_pow]
    exact mul_pos (sq_pos_of_ne_zero (norm_ne_zero_iff.mpr hiε)) (sq_pos_of_ne_zero hp)
  · apply Filter.Tendsto.congr' (Filter.EventuallyEq.symm hratio)
    exact (hlim.norm.pow 2).div_const _

end ModernODE
