import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Basic
import Mathlib.Tactic

/-!
# Extended directional norms and PL constants

These definitions keep the appendix's convention that an unbounded first
variation has infinite L² norm. They do not assert that a trajectory residual
has a particular first variation: that is a separate analytic obligation.
-/

noncomputable section

namespace ModernODE

variable {S V H : Type*} [SeminormedAddCommGroup V] [SeminormedAddCommGroup H]

/-- A finite upper bound for the norm of a first variation. No assumption of
existence of an ambient-space gradient is made. -/
def BoundsVariation (d : V → ℝ) (G : ℝ) : Prop :=
  0 ≤ G ∧ ∀ v, |d v| ≤ G * ‖v‖

/-- The squared-residual loss, with the reference minimum equal to zero. -/
def residualLoss (r : H) : ℝ := ‖r‖ ^ 2 / 2

/-- A nonnegative PL constant in the extended directional-norm sense.
For each trajectory, every finite upper bound `G` on its first variation
must satisfy `μ * ‖r‖² ≤ G²`. When no finite bound exists, the inequality
holds automatically, matching an infinite gradient norm. -/
def IsPL (r : S → H) (d : S → V → ℝ) (μ : ℝ) : Prop :=
  0 ≤ μ ∧ ∀ s G, BoundsVariation (d s) G → μ * ‖r s‖ ^ 2 ≤ G ^ 2

 theorem zero_isPL (r : S → H) (d : S → V → ℝ) : IsPL r d 0 := by
  constructor
  · exact le_rfl
  · intro s G _
    simp only [zero_mul]
    exact sq_nonneg G

 theorem IsPL.mono {r : S → H} {d : S → V → ℝ} {μ ν : ℝ}
    (hμ : IsPL r d μ) (hν : 0 ≤ ν) (hle : ν ≤ μ) : IsPL r d ν := by
  refine ⟨hν, ?_⟩
  intro s G hG
  exact (mul_le_mul_of_nonneg_right hle (sq_nonneg ‖r s‖)).trans (hμ.2 s G hG)

 theorem isPL_iff_loss (r : S → H) (d : S → V → ℝ) (μ : ℝ) :
    IsPL r d μ ↔ 0 ≤ μ ∧ ∀ s G, BoundsVariation (d s) G →
      μ * residualLoss (r s) ≤ G ^ 2 / 2 := by
  unfold IsPL residualLoss
  constructor
  · rintro ⟨hμ, h⟩
    refine ⟨hμ, ?_⟩
    intro s G hG
    have := h s G hG
    nlinarith
  · rintro ⟨hμ, h⟩
    refine ⟨hμ, ?_⟩
    intro s G hG
    have := h s G hG
    nlinarith

/-- A real-valued supremum needs boundedness; the stable theorem below cannot
silently assume Lean's real `sSup` represents an infinite optimum. -/
def optimalPL (r : S → H) (d : S → V → ℝ) : ℝ := sSup {μ | IsPL r d μ}

 theorem le_optimalPL {r : S → H} {d : S → V → ℝ} {μ : ℝ}
    (hb : BddAbove {ν | IsPL r d ν}) (hμ : IsPL r d μ) : μ ≤ optimalPL r d := by
  exact le_csSup hb hμ

 theorem optimalPL_le {r : S → H} {d : S → V → ℝ} {C : ℝ}
    (hC : ∀ μ, IsPL r d μ → μ ≤ C) : optimalPL r d ≤ C := by
  exact csSup_le ⟨0, zero_isPL r d⟩ (fun μ hμ => hC μ hμ)

 theorem isPL_le_ratio {r : S → H} {d : S → V → ℝ} {μ : ℝ}
    (hμ : IsPL r d μ) (s : S) (G : ℝ) (hG : BoundsVariation (d s) G)
    (hr : 0 < ‖r s‖ ^ 2) : μ ≤ G ^ 2 / ‖r s‖ ^ 2 := by
  exact (le_div_iff₀ hr).2 (hμ.2 s G hG)

/-- One nonzero residual with finite variation norm bounds all PL constants. -/
 theorem plConstants_bddAbove {r : S → H} {d : S → V → ℝ}
    (s : S) (G : ℝ) (hG : BoundsVariation (d s) G)
    (hr : 0 < ‖r s‖ ^ 2) : BddAbove {μ | IsPL r d μ} := by
  exact ⟨G ^ 2 / ‖r s‖ ^ 2, fun μ hμ => isPL_le_ratio hμ s G hG hr⟩

/-- With a bounded set of admissible constants, the real optimum is nonnegative. -/
theorem optimalPL_nonneg {r : S → H} {d : S → V → ℝ}
    (hb : BddAbove {μ | IsPL r d μ}) : 0 ≤ optimalPL r d :=
  le_optimalPL hb (zero_isPL r d)

/-- The supremum is itself an admissible PL constant when finite. -/
theorem optimalPL_isPL {r : S → H} {d : S → V → ℝ}
    (hb : BddAbove {μ | IsPL r d μ}) : IsPL r d (optimalPL r d) := by
  refine ⟨optimalPL_nonneg hb, ?_⟩
  intro s G hG
  by_cases hz : ‖r s‖ ^ 2 = 0
  · simp only [hz, mul_zero]
    exact sq_nonneg G
  · have hr : 0 < ‖r s‖ ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm hz)
    have h := optimalPL_le (fun μ hμ => isPL_le_ratio hμ s G hG hr)
    exact (le_div_iff₀ hr).mp h

end ModernODE
