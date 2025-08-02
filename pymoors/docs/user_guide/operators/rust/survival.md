
The survival operator follows the same logic than selection operator, in that each pre‑defined algorithm in `moors` defines exactly one selection operator. For example, the `NSGA-II` algorithm uses a *ranking‑by‑crowding‑distance* survival operator, while `NSGA-III` uses a reference points based operator. The user can only provide their own survival operator to a custom algorithm—not to the algorithms that come pre‑defined in moors.

A survival operator in `moors` is any type that implements the {{ docs_rs("trait", "operators.SurvivalOperator") }} trait. For example:

```Rust
use crate::genetic::{D01, IndividualMOO};
use crate::operators::selection::{DuelResult, SelectionOperator};
use crate::random::RandomGenerator;

#[derive(Debug, Clone)]
pub struct RandomSelection;

impl SelectionOperator for RandomSelection {
    type FDim = ndarray::Ix2;

    fn tournament_duel<'a, ConstrDim>(
        &self,
        p1: &IndividualMOO<'a, ConstrDim>,
        p2: &IndividualMOO<'a, ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) -> DuelResult
    where
        ConstrDim: D01,
    {
        if let result @ DuelResult::LeftWins | result @ DuelResult::RightWins =
            Self::feasibility_dominates(p1, p2)
        {
            return result;
        }
        // Otherwise, both are feasible or both are infeasible => random winner.
        if rng.gen_bool(0.5) {
            DuelResult::LeftWins
        } else {
            DuelResult::RightWins
        }
    }
}
```

Note that we have defined an associated type `type FDim = ndarray::Ix2`, this is because, in this example, this operator will be used for a multi‑objective algorithm. The selection operators defined in moors must specify the fitness dimension. Note that this is the selection operator used by the NSGA‑III algorithm: it performs a random selection that gives priority to feasibility, which is why we use the trait’s static method `Self::feasibility_dominates`.
