//! # Particle Swarm Optimization engine
//!
//! [`Pso`] is the PSO counterpart of [`crate::algorithms::GeneticAlgorithm`]:
//! same overall shape (builder-configured engine exposing `run()` +
//! `population`), but the per-iteration update rule is fundamentally
//! different. A GA discards and rebuilds its population every generation via
//! crossover/mutation; a PSO swarm keeps a **fixed identity per particle**
//! and moves each one according to its own memory (`pbest`) and the swarm's
//! collective memory (`gbest`), see [`Swarm`].
//!
//! This module only defines the generic engine (pluggable via
//! [`VelocityUpdateOperator`] and [`MeritOperator`]). Concrete, ready-to-use
//! PSO variants (e.g. MO-ETPSO) live under [`crate::algorithms::pso`].
use ndarray::{Array1, Array2, Axis};

use crate::{
    algorithms::helpers::{AlgorithmContext, AlgorithmError},
    evaluator::{ConstraintsFn, Evaluator, FitnessFn},
    genetic::{Constraints, D12, Fitness, PopulationMOO},
    helpers::printer::algorithm_printer,
    operators::{
        SamplingOperator,
        pso::{MeritOperator, VelocityUpdateOperator},
        sampling::RandomSamplingFloat,
    },
    random::{MOORandomGenerator, RandomGenerator},
};

/// Full state of a PSO swarm.
///
/// Unlike a GA `Population` (rebuilt from scratch every generation via
/// crossover/mutation), a swarm keeps a **fixed identity per particle**: row
/// `i` of every array below always refers to the same particle across the
/// whole run, which is what lets `velocities` and `pbest_*` make sense as
/// "memory" a particle carries with it over iterations.
#[derive(Debug, Clone)]
pub struct Swarm<ConstrDim>
where
    ConstrDim: D12,
{
    /// Current positions (`genes`), fitness and constraints of every particle.
    pub population: PopulationMOO<ConstrDim>,
    pub velocities: Array2<f64>,
    pub pbest_genes: Array2<f64>,
    pub pbest_fitness: Fitness<ndarray::Ix2>,
    pub pbest_constraints: Constraints<ConstrDim>,
    pub pbest_merit: Array1<f64>,
    /// `1 x num_vars` row: the best position found by any particle so far.
    pub gbest_genes: Array2<f64>,
    pub gbest_merit: f64,
}

impl<ConstrDim> Swarm<ConstrDim>
where
    ConstrDim: D12,
{
    /// Builds the initial swarm: velocities start at zero, and every
    /// particle's personal best is its own initial (evaluated) position.
    pub fn init(population: PopulationMOO<ConstrDim>, merit: &impl MeritOperator) -> Self {
        let n = population.len();
        let d = population.genes.ncols();
        let velocities = Array2::<f64>::zeros((n, d));
        let merit_scores = merit.compute(&population);

        let pbest_genes = population.genes.clone();
        let pbest_fitness = population.fitness.clone();
        let pbest_constraints = population.constraints.clone();
        let pbest_merit = merit_scores.clone();

        let gbest_idx = argmax(&merit_scores);
        let gbest_genes = population.genes.select(Axis(0), &[gbest_idx]);
        let gbest_merit = merit_scores[gbest_idx];

        Self {
            population,
            velocities,
            pbest_genes,
            pbest_fitness,
            pbest_constraints,
            pbest_merit,
            gbest_genes,
            gbest_merit,
        }
    }

    /// Replaces positions/fitness/constraints with a freshly evaluated
    /// population that has the **same particle order** as before, updates
    /// each particle's `pbest` if its new position is an improvement, and
    /// refreshes `gbest` from the updated `pbest` archive (swarm elitism).
    ///
    /// Merit scores are only meaningful when compared within a single
    /// non-dominated-sorting/crowding computation (rank 0 in a small front
    /// isn't comparable to rank 0 in a much larger one). So, instead of
    /// separately scoring `new_population` and comparing against a merit
    /// value cached from a *previous* (differently-shaped) front, this
    /// builds one population out of the current `pbest` archive plus the
    /// new positions and scores everything at once — analogous to how
    /// NSGA-II ranks parents+offspring together before selecting survivors.
    pub fn advance(
        &mut self,
        new_population: PopulationMOO<ConstrDim>,
        merit: &impl MeritOperator,
    ) {
        let n = new_population.len();
        let pbest_population = self.pbest_population();
        let combined = PopulationMOO::merge(&pbest_population, &new_population);
        let merit_scores = merit.compute(&combined);

        for i in 0..n {
            let pbest_score = merit_scores[i];
            let new_score = merit_scores[n + i];
            if new_score > pbest_score {
                self.pbest_merit[i] = new_score;
                self.pbest_genes
                    .row_mut(i)
                    .assign(&new_population.genes.row(i));
                self.pbest_fitness
                    .row_mut(i)
                    .assign(&new_population.fitness.row(i));
                self.pbest_constraints
                    .index_axis_mut(Axis(0), i)
                    .assign(&new_population.constraints.index_axis(Axis(0), i));
            } else {
                self.pbest_merit[i] = pbest_score;
            }
        }

        let gbest_idx = argmax(&self.pbest_merit);
        self.gbest_merit = self.pbest_merit[gbest_idx];
        self.gbest_genes = self.pbest_genes.select(Axis(0), &[gbest_idx]);

        self.population = new_population;
    }

    /// Reinitializes the positions of the given particle indices (and resets
    /// their velocity to zero). Used by the stagnation/randomization
    /// mechanism to help the swarm escape a stalled Pareto front.
    pub fn reinit_positions(&mut self, indices: &[usize], new_positions: &Array2<f64>) {
        for (row, &idx) in new_positions.outer_iter().zip(indices.iter()) {
            self.population.genes.row_mut(idx).assign(&row);
            self.velocities.row_mut(idx).fill(0.0);
        }
    }
    /// Builds a `Population` snapshot of the swarm's memory: every
    /// particle's personal best (`pbest`) position, fitness and constraints.
    ///
    /// This — not the raw current positions — is the meaningful "result" of
    /// a PSO run: particles keep moving (and can transiently overshoot past
    /// good positions) for as long as the algorithm runs, while `pbest` only
    /// ever improves.
    pub fn pbest_population(&self) -> PopulationMOO<ConstrDim> {
        PopulationMOO::new(
            self.pbest_genes.clone(),
            self.pbest_fitness.clone(),
            self.pbest_constraints.clone(),
        )
    }
}

fn argmax(values: &Array1<f64>) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .expect("merit array must not be empty")
}

#[derive(Debug)]
pub struct Pso<V, M, F, G>
where
    V: VelocityUpdateOperator,
    M: MeritOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
{
    pub swarm: Option<Swarm<G::Dim>>,
    velocity_update: V,
    merit: M,
    sampler: RandomSamplingFloat,
    evaluator: Evaluator<F, G>,
    pub context: AlgorithmContext,
    lower_bound: f64,
    upper_bound: f64,
    stagnation_threshold: usize,
    stagnation_counter: usize,
    verbose: bool,
    rng: MOORandomGenerator,
}

impl<V, M, F, G> Pso<V, M, F, G>
where
    V: VelocityUpdateOperator,
    M: MeritOperator,
    F: FitnessFn<Dim = ndarray::Ix2>,
    G: ConstraintsFn,
{
    pub fn new(
        velocity_update: V,
        merit: M,
        sampler: RandomSamplingFloat,
        evaluator: Evaluator<F, G>,
        context: AlgorithmContext,
        lower_bound: f64,
        upper_bound: f64,
        stagnation_threshold: usize,
        verbose: bool,
        rng: MOORandomGenerator,
    ) -> Self {
        Self {
            swarm: None,
            velocity_update,
            merit,
            sampler,
            evaluator,
            context,
            lower_bound,
            upper_bound,
            stagnation_threshold,
            stagnation_counter: 0,
            verbose,
            rng,
        }
    }

    /// Returns the current swarm's population (positions/fitness/constraints),
    /// or `None` before `run()` has been called.
    pub fn population(&self) -> Option<&PopulationMOO<G::Dim>> {
        self.swarm.as_ref().map(|s| &s.population)
    }

    /// Returns the swarm's elitist memory (every particle's personal best),
    /// or `None` before `run()` has been called. This is the recommended
    /// way to read out the result of a PSO run (see [`Swarm::pbest_population`]).
    pub fn best_population(&self) -> Option<PopulationMOO<G::Dim>> {
        self.swarm.as_ref().map(|s| s.pbest_population())
    }

    fn clamp_to_bounds(&self, positions: &mut Array2<f64>) {
        positions.mapv_inplace(|x| x.clamp(self.lower_bound, self.upper_bound));
    }

    fn next(&mut self) -> Result<(), AlgorithmError> {
        let swarm = self.swarm.as_ref().expect("swarm must be initialized");

        let new_velocities = self.velocity_update.operate(
            &swarm.population.genes,
            &swarm.velocities,
            &swarm.pbest_genes,
            &swarm.gbest_genes,
            &mut self.rng,
        );
        let mut new_positions = &swarm.population.genes + &new_velocities;
        self.clamp_to_bounds(&mut new_positions);

        let evaluated_population = self.evaluator.evaluate(new_positions)?;

        let swarm = self.swarm.as_mut().expect("swarm must be initialized");
        swarm.velocities = new_velocities;
        let previous_gbest_merit = swarm.gbest_merit;
        swarm.advance(evaluated_population, &self.merit);

        // Stagnation tracking: if the swarm's global best did not improve
        // this iteration, count it towards a randomization phase that helps
        // escape a stalled Pareto front (MO-ETPSO, section 2.10).
        if swarm.gbest_merit <= previous_gbest_merit {
            self.stagnation_counter += 1;
        } else {
            self.stagnation_counter = 0;
        }

        if self.stagnation_counter >= self.stagnation_threshold {
            self.randomize_stagnated_particles();
            self.stagnation_counter = 0;
        }

        Ok(())
    }

    /// Reinitializes a random subset of particles within the search bounds.
    /// The subset size grows with how long the swarm has been stagnated,
    /// capped at the full swarm size.
    fn randomize_stagnated_particles(&mut self) {
        let swarm = self.swarm.as_mut().expect("swarm must be initialized");
        let n = swarm.population.len();
        let num_to_randomize = self.stagnation_threshold.min(n).max(1);

        let mut indices: Vec<usize> = (0..n).collect();
        self.rng.shuffle_vec_usize(&mut indices);
        let chosen = &indices[..num_to_randomize];

        let new_positions =
            self.sampler
                .operate(chosen.len(), self.context.num_vars, &mut self.rng);
        swarm.reinit_positions(chosen, &new_positions);
    }

    pub fn run(&mut self) -> Result<(), AlgorithmError> {
        let initial_positions = self.sampler.operate(
            self.context.population_size,
            self.context.num_vars,
            &mut self.rng,
        );
        let initial_population = self.evaluator.evaluate(initial_positions)?;
        self.swarm = Some(Swarm::init(initial_population, &self.merit));

        for current_iter in 0..self.context.num_iterations {
            self.next()?;
            self.context.set_current_iteration(current_iter);
            if self.verbose {
                algorithm_printer(
                    &self.swarm.as_ref().unwrap().population.fitness,
                    current_iter + 1,
                );
            }
        }
        Ok(())
    }
}
