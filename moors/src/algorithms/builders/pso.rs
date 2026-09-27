//! Builder for the generic [`Pso`] engine.
//!
//! This is the low-level, fully-generic builder (it takes the velocity-update
//! and merit operators directly). Concrete algorithms such as
//! [`MoEtpso`](crate::algorithms::MoEtpso) wrap this builder to expose only
//! their own tunable parameters instead of raw operator structs — mirroring
//! how `Nsga2Builder` wraps [`AlgorithmBuilder`](crate::algorithms::AlgorithmBuilder)
//! for the GA family.
use derive_builder::Builder;

use crate::{
    algorithms::Pso,
    algorithms::helpers::AlgorithmContextBuilder,
    evaluator::{ConstraintsFn, EvaluatorBuilder, FitnessFn},
    operators::{
        pso::{MeritOperator, VelocityUpdateOperator},
        sampling::RandomSamplingFloat,
    },
    random::MOORandomGenerator,
};

#[derive(Builder, Debug)]
#[builder(
    pattern = "owned",
    name = "PsoBuilder",
    build_fn(name = "build_params", validate = "Self::validate")
)]
pub struct PsoParams<F, G, V, M>
where
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    V: VelocityUpdateOperator,
    M: MeritOperator,
{
    fitness_fn: F,
    constraints_fn: G,
    velocity_update: V,
    merit: M,
    num_vars: usize,
    population_size: usize,
    num_iterations: usize,
    lower_bound: f64,
    upper_bound: f64,
    #[builder(default = "20")]
    stagnation_threshold: usize,
    #[builder(default = "true")]
    keep_infeasible: bool,
    #[builder(default = "false")]
    verbose: bool,
    #[builder(setter(strip_option), default = "None")]
    seed: Option<u64>,
}

impl<F, G, V, M> PsoBuilder<F, G, V, M>
where
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
    V: VelocityUpdateOperator,
    M: MeritOperator,
{
    fn validate(&self) -> Result<(), PsoBuilderError> {
        if let Some(num_vars) = self.num_vars {
            if num_vars == 0 {
                return Err(PsoBuilderError::ValidationError(
                    "Number of variables must be greater than 0".to_string(),
                ));
            }
        }
        if let Some(population_size) = self.population_size {
            if population_size == 0 {
                return Err(PsoBuilderError::ValidationError(
                    "Population size must be greater than 0".to_string(),
                ));
            }
        }
        if let Some(num_iterations) = self.num_iterations {
            if num_iterations == 0 {
                return Err(PsoBuilderError::ValidationError(
                    "Number of iterations must be greater than 0".to_string(),
                ));
            }
        }
        if let Some(stagnation_threshold) = self.stagnation_threshold {
            if stagnation_threshold == 0 {
                return Err(PsoBuilderError::ValidationError(
                    "Stagnation threshold must be greater than 0".to_string(),
                ));
            }
        }
        if let (Some(lower_bound), Some(upper_bound)) = (self.lower_bound, self.upper_bound) {
            if lower_bound >= upper_bound {
                return Err(PsoBuilderError::ValidationError(format!(
                    "Lower bound ({}) must be less than upper bound ({})",
                    lower_bound, upper_bound
                )));
            }
        }
        Ok(())
    }

    pub fn build(self) -> Result<Pso<V, M, F, G>, PsoBuilderError> {
        let params = self.build_params()?;

        let evaluator = EvaluatorBuilder::default()
            .fitness(params.fitness_fn)
            .constraints(params.constraints_fn)
            .keep_infeasible(params.keep_infeasible)
            .build()
            .expect("Params already validated in build_params");

        let context = AlgorithmContextBuilder::default()
            .num_vars(params.num_vars)
            .population_size(params.population_size)
            .num_offsprings(0)
            .num_iterations(params.num_iterations)
            .lower_bound(Some(params.lower_bound))
            .upper_bound(Some(params.upper_bound))
            .build()
            .expect("Params already validated in build_params");

        let sampler = RandomSamplingFloat::new(params.lower_bound, params.upper_bound);
        let rng = MOORandomGenerator::new_from_seed(params.seed);

        Ok(Pso::new(
            params.velocity_update,
            params.merit,
            sampler,
            evaluator,
            context,
            params.lower_bound,
            params.upper_bound,
            params.stagnation_threshold,
            params.verbose,
            rng,
        ))
    }
}
