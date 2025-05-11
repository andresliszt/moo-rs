use crate::{
    algorithms::helpers::context::AlgorithmContext,
    genetic::{Fronts, FrontsExt, Population},
    non_dominated_sorting::build_fronts,
    operators::GeneticOperator,
    random::RandomGenerator,
};

pub mod agemoea;
mod helpers;
pub mod nsga2;
pub mod nsga3;
pub mod reference_points;
pub mod revea;
pub mod rnsga2;
pub mod spea2;

pub use agemoea::AgeMoeaSurvival;
pub use nsga2::Nsga2RankCrowdingSurvival;
pub use nsga3::Nsga3ReferencePoints;
pub use revea::ReveaReferencePointsSurvival;
pub use rnsga2::Rnsga2ReferencePointsSurvival;

/// Controls how the diversity (crowding) metric is compared during tournament selection.
#[derive(Debug, Clone)]
pub enum SurvivalScoringComparison {
    /// Larger survival scoring (e.g crowding sitance) is preferred.
    Maximize,
    /// Smaller survival scoring (crowding) metric is preferred.
    Minimize,
}

/// The base trait for **all** survival operators.
///
/// A survival operator takes a full `Population` and
/// returns the `num_survive` individuals that will move on to the next generation.
/// Algorithms that need custom survival logic (e.g. NSGA3’s reference-point logic)
/// implement this trait directly.
pub trait SurvivalOperator: GeneticOperator {
    /// Selects the individuals that will survive to the next generation.
    fn operate(
        &mut self,
        population: Population,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
        algorithm_context: &AlgorithmContext,
    ) -> Population;
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
pub trait FrontsAndRankingBasedSurvival: SurvivalOperator {
    /// Returns whether the survival scoring should be maximized or minimized.
    fn scoring_comparison(&self) -> SurvivalScoringComparison {
        SurvivalScoringComparison::Maximize
    }

    /// Computes the survival score for a given front's fitness.
    /// This is the only method that needs to be overridden by each survival operator.
    fn set_front_survival_score(
        &self,
        fronts: &mut Fronts,
        rng: &mut impl RandomGenerator,
        algorithm_context: &AlgorithmContext,
    );

    /// Selects the individuals that will survive to the next generation.
    /// Default `operate` that builds fronts, scores, and splits any "overflowing" front.
    fn operate(
        &mut self,
        population: Population,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
        algorithm_context: &AlgorithmContext,
    ) -> Population {
        // Build fronts
        let mut fronts = build_fronts(population, num_survive);
        // Set survival score
        self.set_front_survival_score(&mut fronts, rng, algorithm_context);
        // Drain all fronts.
        let drained = fronts.drain(..);
        let mut survivors_parts: Vec<Population> = Vec::new();
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
    fn operate(
        &mut self,
        population: Population,
        num_survive: usize,
        rng: &mut impl RandomGenerator,
        algorithm_context: &AlgorithmContext,
    ) -> Population {
        // Delegate to the FrontsAndRankingBasedSurvival default implementation
        <T as FrontsAndRankingBasedSurvival>::operate(
            self,
            population,
            num_survive,
            rng,
            algorithm_context,
        )
    }
}
