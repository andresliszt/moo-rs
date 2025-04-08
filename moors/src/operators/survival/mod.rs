use crate::{
    algorithms::AlgorithmContext,
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

/// Controls how the diversity (crowding) metric is compared during tournament selection.
#[derive(Clone, Debug)]
pub enum SurvivalScoringComparison {
    /// Larger survival scoring (e.g crowding sitance) is preferred.
    Maximize,
    /// Smaller survival scoring (crowding) metric is preferred.
    Minimize,
}

/// The SurvivalOperator trait extends GeneticOperator and requires that concrete operators
/// provide a method for computing the survival score from a front's fitness.
pub trait SurvivalOperator: GeneticOperator {
    /// Returns whether the survival scoring should be maximized or minimized.
    fn scoring_comparison(&self) -> SurvivalScoringComparison {
        SurvivalScoringComparison::Maximize
    }

    /// Computes the survival score for a given front's fitness.
    /// This is the only method that needs to be overridden by each survival operator.
    fn set_survival_score(
        &self,
        fronts: &mut Fronts,
        rng: &mut dyn RandomGenerator,
        algorithm_context: &AlgorithmContext,
    );

    /// Selects the individuals that will survive to the next generation.
    /// The default implementation uses the survival score to select individuals.
    fn operate(
        &mut self,
        population: Population,
        n_survive: usize,
        rng: &mut dyn RandomGenerator,
        algorithm_context: &AlgorithmContext,
    ) -> Population {
        // Build fronts
        let mut fronts = build_fronts(population, n_survive);
        // Set survival score
        self.set_survival_score(&mut fronts, rng, algorithm_context);
        // Drain all fronts.
        let drained = fronts.drain(..);
        let mut survivors_parts: Vec<Population> = Vec::new();
        let mut n_survivors = 0;

        for front in drained {
            let front_len = front.len();
            if n_survivors + front_len <= n_survive {
                // The entire front fits.
                survivors_parts.push(front);
                n_survivors += front_len;
            } else {
                // Splitting front: only part of the front is needed.
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // Use Splitting context regardless of i.
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
