use crate::{
    genetic::{D12, Fronts, FrontsExt, PopulationMOO},
    non_dominated_sorting::build_fronts,
    operators::survival::SurvivalOperator,
    random::RandomGenerator,
};

pub(crate) mod agemoea;
pub(crate) mod helpers;
pub(crate) mod nsga2;
pub(crate) mod nsga3;
pub(crate) mod reference_points;
pub(crate) mod revea;
pub(crate) mod rnsga2;
pub(crate) mod spea2;

pub use agemoea::AgeMoeaSurvival;
pub use nsga2::Nsga2RankCrowdingSurvival;
pub use nsga3::{Nsga3ReferencePoints, Nsga3ReferencePointsSurvival};
pub use reference_points::{
    DanAndDenisReferencePoints, NormalBoundaryDivisions, StructuredReferencePoints,
};
pub use revea::ReveaReferencePointsSurvival;
pub use rnsga2::Rnsga2ReferencePointsSurvival;
pub use spea2::Spea2KnnSurvival;

/// Controls how the diversity (crowding) metric is compared during tournament selection.
#[derive(Debug, Clone)]
pub enum SurvivalScoringComparison {
    /// Larger survival scoring (e.g crowding sitance) is preferred.
    Maximize,
    /// Smaller survival scoring (crowding) metric is preferred.
    Minimize,
}

/// A more specific survival trait for **front-and-ranking** algorithms
/// (e.g. NSGA-II, AgeMoea, R-NSGA-II) that:
/// 1. Partition into non-dominated fronts,
/// 2. Assign a per-front diversity score (e.g. crowding distance),
/// 3. Fill full fronts until the next would overflow,
/// 4. Split that “overflowing” front by highest diversity.
///
/// For these algorithms, you only need to implement
/// `set_front_survival_score` to compute each front’s scores;
/// the default `operate` covers the rest.
pub trait FrontsAndRankingBasedSurvival: SurvivalOperator<FDim = ndarray::Ix2> {
    /// Returns whether the survival scoring should be maximized or minimized.
    fn scoring_comparison(&self) -> SurvivalScoringComparison {
        SurvivalScoringComparison::Maximize
    }

    /// Computes the survival score for a given front's fitness.
    /// This is the only method that needs to be overridden by each survival operator.
    fn set_front_survival_score<ConstrDim>(
        &self,
        fronts: &mut Fronts<ConstrDim>,
        rng: &mut impl RandomGenerator,
    ) where
        ConstrDim: D12;

    /// Selects the individuals that will survive to the next generation.
    /// Default `operate` that builds fronts, scores, and splits any "overflowing" front.
    fn operate<ConstrDim>(
        &mut self,
        population: PopulationMOO<ConstrDim>,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
    ) -> PopulationMOO<ConstrDim>
    where
        ConstrDim: D12,
    {
        // Build fronts
        let mut fronts = build_fronts(population, num_survive);
        // Set survival score
        self.set_front_survival_score(&mut fronts, rng);
        // Drain all fronts.
        let drained = fronts.drain(..);
        let mut survivors_parts: Vec<PopulationMOO<ConstrDim>> = Vec::new();
        let mut n_survivors = 0;

        for front in drained {
            let front_len = front.len();
            if n_survivors + front_len <= num_survive {
                // The entire front fits.
                survivors_parts.push(front);
                n_survivors += front_len;
            } else {
                // Splitting front: only part of the front is needed.
                let remaining = num_survive - n_survivors;
                if remaining > 0 {
                    // Clone survival_score vector for sorting.
                    let scores = front
                        .survival_score
                        .clone()
                        .expect("No survival score set for splitting front");
                    // Get indices for the current front.
                    let mut indices: Vec<usize> = (0..front_len).collect();
                    indices.sort_by(|&i, &j| match self.scoring_comparison() {
                        SurvivalScoringComparison::Maximize => scores[j]
                            .partial_cmp(&scores[i])
                            .unwrap_or(std::cmp::Ordering::Equal),
                        SurvivalScoringComparison::Minimize => scores[i]
                            .partial_cmp(&scores[j])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    });
                    // Select exactly the required number of individuals.
                    let selected_indices: Vec<usize> =
                        indices.into_iter().take(remaining).collect();
                    let partial = front.selected(&selected_indices);
                    survivors_parts.push(partial);
                }
                break;
            }
        }
        survivors_parts.to_population()
    }
}

// For any T that implements FrontsAndRankingBasedSurvival, also implement SurvivalOperator
impl<T: FrontsAndRankingBasedSurvival> SurvivalOperator for T {
    type FDim = ndarray::Ix2;

    fn operate<ConstrDim>(
        &mut self,
        population: PopulationMOO<ConstrDim>,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
    ) -> PopulationMOO<ConstrDim>
    where
        ConstrDim: D12,
    {
        // Delegate to the FrontsAndRankingBasedSurvival default implementation
        <T as FrontsAndRankingBasedSurvival>::operate(self, population, num_survive, rng)
    }
}
