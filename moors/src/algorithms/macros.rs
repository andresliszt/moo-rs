#[macro_export]
macro_rules! create_algorithm {
    ($(#[$meta:meta])* $algorithm:ident, $selector:ty, $survivor:ty) => {
        use $crate::{
            algorithms::{AlgorithmBuilder, AlgorithmBuilderError, AlgorithmError, GeneticAlgorithm},
            duplicates::PopulationCleaner,
            evaluator::{ConstraintsFn, FitnessFn},
            operators::{
                CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator, SurvivalOperator,
            },
        };

        $(#[$meta])*
        #[derive(Debug)]
        pub struct $algorithm<S, Cross, Mut, F, G, DC>
        where
            S: SamplingOperator,
            $selector: SelectionOperator<FDim = F::Dim>,
            $survivor: SurvivalOperator<FDim = F::Dim>,
            Cross: CrossoverOperator,
            Mut: MutationOperator,
            F: FitnessFn,
            G: ConstraintsFn,
            DC: PopulationCleaner,
        {
            pub inner: GeneticAlgorithm<
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

        impl<S, Cross, Mut, F, G, DC> $algorithm<S, Cross, Mut, F, G, DC>
        where
            S: SamplingOperator,
            $selector: SelectionOperator<FDim = F::Dim>,
            $survivor: SurvivalOperator<FDim = F::Dim>,
            Cross: CrossoverOperator,
            Mut: MutationOperator,
            F: FitnessFn,
            G: ConstraintsFn,
            DC: PopulationCleaner
        {
            pub fn run(&mut self) -> Result<(), AlgorithmError> {
                self.inner.run()
            }

            pub fn next_pop(&mut self) -> Result<(), AlgorithmError> {
                self.inner.next_pop()
            }

            pub fn initialize(&mut self) -> Result<(), AlgorithmError> {
                self.inner.initialize()
            }

            pub fn set_current_iteration(&mut self, current_iter: usize) {
                self.inner.set_current_iteration(current_iter);
            }

            /// Delegate `population` to the inner algorithm
            pub fn population(
                &self,
            ) -> Result<&$crate::genetic::Population<F::Dim, G::Dim>, $crate::algorithms::AlgorithmError> {
                match &self.inner.population {
                    Some(v) => Ok(v),
                    None => Err($crate::algorithms::AlgorithmError::Initialization(
                        $crate::algorithms::InitializationError::NotInitializated(
                            "population is not set".into(),
                        ),
                    )),
                }
            }
        }

        paste! {
            pub struct [<$algorithm Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: SamplingOperator,
                $selector: SelectionOperator<FDim = F::Dim>,
                $survivor: SurvivalOperator<FDim = F::Dim>,
                Cross: CrossoverOperator,
                Mut: MutationOperator,
                F: FitnessFn,
                G: ConstraintsFn,
                DC: PopulationCleaner,
            {
                inner_builder: AlgorithmBuilder<
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

            impl<S, Cross, Mut, F, G, DC> [<$algorithm Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: SamplingOperator,
                $selector: SelectionOperator<FDim = F::Dim>,
                $survivor: SurvivalOperator<FDim = F::Dim>,
                Cross: CrossoverOperator,
                Mut: MutationOperator,
                F: FitnessFn,
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
                pub fn context_id(mut self, v: usize) -> Self { self.inner_builder = self.inner_builder.context_id(v); self }

                pub fn build(self) -> Result<$algorithm<S, Cross, Mut, F, G, DC>, AlgorithmBuilderError> {
                    Ok($algorithm {
                        inner: self.inner_builder.build()?,
                    })
                }
            }
        }
    };
}
