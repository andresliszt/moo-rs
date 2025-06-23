macro_rules! create_algorithm {
    ($(#[$meta:meta])* $algo:ident, $selector:ty, $survivor:ty) => {
        use crate::{
            algorithms::moo::{GeneticAlgorithmMOO, AlgorithmMOOBuilder, AlgorithmMOOBuilderError, AlgorithmError},
            duplicates::PopulationCleaner,
            evaluator::{ConstraintsFn, FitnessFn},
            operators::{
                CrossoverOperator, MutationOperator, SamplingOperator,
            },
        };

        $(#[$meta])*
        #[derive(Debug)]
        pub struct $algo<S, Cross, Mut, F, G, DC>
        where
            S: SamplingOperator,
            Cross: CrossoverOperator,
            Mut: MutationOperator,
            F: FitnessFn<Dim = ndarray::Ix2>,
            G: ConstraintsFn,
            DC: PopulationCleaner,
        {
            pub inner: GeneticAlgorithmMOO<
                S,
                $selector,
                $survivor,
                Cross,
                Mut,
                F,
                G,
                DC,
            >,
        }

        impl<S, Cross, Mut, F, G, DC> $algo<S, Cross, Mut, F, G, DC>
        where
            S: SamplingOperator,
            Cross: CrossoverOperator,
            Mut: MutationOperator,
            F: FitnessFn<Dim = ndarray::Ix2>,
            G: ConstraintsFn,
            DC: PopulationCleaner
        {
            pub fn run(&mut self) -> Result<(), AlgorithmError> {
                self.inner.run()
            }

            /// Delegate `population` to the inner algorithm
            pub fn population(
                &self,
            ) -> Result<&crate::genetic::PopulationMOO<G::Dim>, crate::algorithms::AlgorithmError> {
                match &self.inner.population {
                    Some(v) => Ok(v),
                    None => Err(crate::algorithms::AlgorithmError::Initialization(
                        crate::algorithms::InitializationError::NotInitializated(
                            "population is not set".into(),
                        ),
                    )),
                }
            }
        }

        paste! {
            pub struct [<$algo Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: SamplingOperator,
                Cross: CrossoverOperator,
                Mut: MutationOperator,
                F: FitnessFn<Dim = ndarray::Ix2>,
                G: ConstraintsFn,
                DC: PopulationCleaner,
            {
                inner_builder: AlgorithmMOOBuilder<
                    S,
                    $selector,
                    $survivor,
                    Cross,
                    Mut,
                    F,
                    G,
                    DC,
                >,
            }

            impl<S, Cross, Mut, F, G, DC> [<$algo Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: SamplingOperator,
                Cross: CrossoverOperator,
                Mut: MutationOperator,
                F: FitnessFn<Dim = ndarray::Ix2>,
                G: ConstraintsFn,
                DC: PopulationCleaner,
            {
                pub fn sampler(mut self, v: S) -> Self { self.inner_builder = self.inner_builder.sampler(v); self }
                pub fn crossover(mut self, v: Cross) -> Self { self.inner_builder = self.inner_builder.crossover(v); self }
                pub fn mutation(mut self, v: Mut) -> Self { self.inner_builder = self.inner_builder.mutation(v); self }
                pub fn selector(mut self, v: $selector) -> Self { self.inner_builder = self.inner_builder.selector(v); self }
                pub fn survivor(mut self, v: $survivor) -> Self { self.inner_builder = self.inner_builder.survivor(v); self }
                pub fn fitness_fn(mut self, v: F) -> Self { self.inner_builder = self.inner_builder.fitness_fn(v); self }
                pub fn constraints_fn(mut self, v: G) -> Self { self.inner_builder = self.inner_builder.constraints_fn(v); self }
                pub fn duplicates_cleaner(mut self, v: DC) -> Self { self.inner_builder = self.inner_builder.duplicates_cleaner(v); self }
                pub fn num_vars(mut self, v: usize) -> Self { self.inner_builder = self.inner_builder.num_vars(v); self }
                pub fn population_size(mut self, v: usize) -> Self { self.inner_builder = self.inner_builder.population_size(v); self }
                pub fn num_offsprings(mut self, v: usize) -> Self { self.inner_builder = self.inner_builder.num_offsprings(v); self }
                pub fn num_iterations(mut self, v: usize) -> Self { self.inner_builder = self.inner_builder.num_iterations(v); self }
                pub fn mutation_rate(mut self, v: f64) -> Self { self.inner_builder = self.inner_builder.mutation_rate(v); self }
                pub fn crossover_rate(mut self, v: f64) -> Self { self.inner_builder = self.inner_builder.crossover_rate(v); self }
                pub fn keep_infeasible(mut self, v: bool) -> Self { self.inner_builder = self.inner_builder.keep_infeasible(v); self }
                pub fn verbose(mut self, v: bool) -> Self { self.inner_builder = self.inner_builder.verbose(v); self }
                pub fn seed(mut self, v: u64) -> Self { self.inner_builder = self.inner_builder.seed(v); self }

                pub fn build(self) -> Result<$algo<S, Cross, Mut, F, G, DC>, AlgorithmMOOBuilderError> {
                    Ok($algo {
                        inner: self.inner_builder.build()?,
                    })
                }
            }
        }
    };
}
