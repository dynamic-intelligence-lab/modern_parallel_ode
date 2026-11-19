import ModernODE
import Lean.Util.CollectAxioms

/-!
Audit every project theorem's transitive axiom dependencies. Fail on proof
placeholders, new mathematical axioms, or native decision-procedure axioms.
The three allowed axioms are Lean's standard classical foundations.
-/

open Lean Elab Command

run_cmd do
  let env ← getEnv
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut count := 0
  for (name, info) in env.constants.toList do
    if (`ModernODE).isPrefixOf name then
      if info.isAxiom then
        throwError "Project axiom is forbidden: {name}"
      if Lean.getOriginalConstKind? env name == some .thm then
        count := count + 1
        let axioms ← Lean.collectAxioms name
        for ax in axioms do
          unless allowed.contains ax do
            throwError "Unapproved axiom {ax} in {name}"
  if count == 0 then
    throwError "Audit found no project theorems"
  logInfo m!"PASS: audited {count} project theorem declarations; only propext, Classical.choice, and Quot.sound are permitted."

#print axioms ModernODE.theorem1
#print axioms ModernODE.theorem2
#print axioms ModernODE.theorem1_ode
#print axioms ModernODE.exists_isTransition
#print axioms ModernODE.isTransition_unique
#print axioms ModernODE.exists_finite_slope
#print axioms ModernODE.theorem1_ode'
#print axioms ModernODE.theorem1_negField
#print axioms ModernODE.isTransition_comp
#print axioms ModernODE.exists_adjoint_solution
#print axioms ModernODE.exists_forced_family
#print axioms ModernODE.exists_adjoint_family
#print axioms ModernODE.theorem2_ode
#print axioms ModernODE.theorem2_posField
#print axioms ModernODE.C1Field.ofContDiff
#print axioms ModernODE.optimalPL_tendsto_zero
#print axioms ModernODE.posField_optimalPL_tendsto_zero
