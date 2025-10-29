#[macro_export]
macro_rules! create_algorithm {
    ($(#[$meta:meta])* $algorithm:ident, $selector:ty, $survivor:ty) => {
        use crate::{
            algorithms::{GeneticAlgorithm, AlgorithmBuilder, AlgorithmBuilderError, AlgorithmError},
            duplicates::PopulationCleaner,
            evaluator::{ConstraintsFn, FitnessFn},
            operators::{
                SelectionOperator, SurvivalOperator, CrossoverOperator, MutationOperator, SamplingOperator,
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

            /// Delegate `population` to the inner algorithm
            pub fn population(
                &self,
            ) -> Result<&crate::genetic::Population<F::Dim, G::Dim>, crate::algorithms::AlgorithmError> {
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

                pub fn build(self) -> Result<$algorithm<S, Cross, Mut, F, G, DC>, AlgorithmBuilderError> {
                    Ok($algorithm {
                        inner: self.inner_builder.build()?,
                    })
                }
            }
        }
    };
}

#[macro_export]
macro_rules! forw {
    ($builder:ident, $selector:ty, $survivor:ty) => {
        impl<S, Cross, Mut, F, G, DC> ::core::default::Default for $builder<S, Cross, Mut, F, G, DC>
        where
            S: $crate::operators::SamplingOperator,
            $selector: $crate::operators::SelectionOperator<FDim = F::Dim>,
            $survivor: $crate::operators::SurvivalOperator<FDim = F::Dim>,
            Cross: $crate::operators::CrossoverOperator,
            Mut: $crate::operators::MutationOperator,
            F: $crate::evaluator::FitnessFn,
            G: $crate::evaluator::ConstraintsFn,
            DC: $crate::duplicates::PopulationCleaner,
            $crate::algorithms::AlgorithmBuilder<S, $selector, $survivor, Cross, Mut, F, G, DC>:
                ::core::default::Default,
        {
            fn default() -> Self {
                Self {
                    inner: ::core::default::Default::default(),
                }
            }
        }

        impl<S, Cross, Mut, F, G, DC> $builder<S, Cross, Mut, F, G, DC>
        where
            S: $crate::operators::SamplingOperator,
            $selector: $crate::operators::SelectionOperator<FDim = F::Dim>,
            $survivor: $crate::operators::SurvivalOperator<FDim = F::Dim>,
            Cross: $crate::operators::CrossoverOperator,
            Mut: $crate::operators::MutationOperator,
            F: $crate::evaluator::FitnessFn,
            G: $crate::evaluator::ConstraintsFn,
            DC: $crate::duplicates::PopulationCleaner,
        {
            #[inline]
            pub fn sampler(mut self, v: S) -> Self {
                self.inner = self.inner.sampler(v);
                self
            }
            #[inline]
            pub fn crossover(mut self, v: Cross) -> Self {
                self.inner = self.inner.crossover(v);
                self
            }
            #[inline]
            pub fn mutation(mut self, v: Mut) -> Self {
                self.inner = self.inner.mutation(v);
                self
            }
            #[inline]
            pub fn selector(mut self, v: $selector) -> Self {
                self.inner = self.inner.selector(v);
                self
            }
            #[inline]
            pub fn survivor(mut self, v: $survivor) -> Self {
                self.inner = self.inner.survivor(v);
                self
            }
            #[inline]
            pub fn fitness_fn(mut self, v: F) -> Self {
                self.inner = self.inner.fitness_fn(v);
                self
            }
            #[inline]
            pub fn constraints_fn(mut self, v: G) -> Self {
                self.inner = self.inner.constraints_fn(v);
                self
            }
            #[inline]
            pub fn duplicates_cleaner(mut self, v: DC) -> Self {
                self.inner = self.inner.duplicates_cleaner(v);
                self
            }
            #[inline]
            pub fn num_vars(mut self, v: usize) -> Self {
                self.inner = self.inner.num_vars(v);
                self
            }
            #[inline]
            pub fn population_size(mut self, v: usize) -> Self {
                self.inner = self.inner.population_size(v);
                self
            }
            #[inline]
            pub fn num_offsprings(mut self, v: usize) -> Self {
                self.inner = self.inner.num_offsprings(v);
                self
            }
            #[inline]
            pub fn num_iterations(mut self, v: usize) -> Self {
                self.inner = self.inner.num_iterations(v);
                self
            }
            #[inline]
            pub fn mutation_rate(mut self, v: f64) -> Self {
                self.inner = self.inner.mutation_rate(v);
                self
            }
            #[inline]
            pub fn crossover_rate(mut self, v: f64) -> Self {
                self.inner = self.inner.crossover_rate(v);
                self
            }
            #[inline]
            pub fn keep_infeasible(mut self, v: bool) -> Self {
                self.inner = self.inner.keep_infeasible(v);
                self
            }
            #[inline]
            pub fn verbose(mut self, v: bool) -> Self {
                self.inner = self.inner.verbose(v);
                self
            }
            #[inline]
            pub fn seed(mut self, v: u64) -> Self {
                self.inner = self.inner.seed(v);
                self
            }
        }
    };
}

#[macro_export]
macro_rules! create_algorithm_and_builder {
    // Public API — with `extras = [...]` + explicit override_build flag
    ($(#[$meta:meta])* $algorithm:ident, $selector:ty, $survivor:ty,
        extras = [ $( $ef:ident : $ety:ty ),+ $(,)? ],
        override_build_method = $ov:tt
    ) => {
        $crate::create_algorithm_and_builder! {
            @impl
            ($(#[$meta])*) $algorithm, $selector, $survivor,
            extras: [ $( $ef : $ety ),+ ],
            override_build: $ov
        }
    };

    // Public API — with `extras = [...]` and default override_build = false
    ($(#[$meta:meta])* $algorithm:ident, $selector:ty, $survivor:ty,
        extras = [ $( $ef:ident : $ety:ty ),+ $(,)? ]
    ) => {
        $crate::create_algorithm_and_builder! {
            @impl
            ($(#[$meta])*) $algorithm, $selector, $survivor,
            extras: [ $( $ef : $ety ),+ ],
            override_build: false
        }
    };

    // Public API — without extras + explicit override_build flag
    ($(#[$meta:meta])* $algorithm:ident, $selector:ty, $survivor:ty,
        override_build_method = $ov:tt
    ) => {
        $crate::create_algorithm_and_builder! {
            @impl
            ($(#[$meta])*) $algorithm, $selector, $survivor,
            extras: [ ],
            override_build: $ov
        }
    };

    // Public API — without extras, default override_build = false
    ($(#[$meta:meta])* $algorithm:ident, $selector:ty, $survivor:ty) => {
        $crate::create_algorithm_and_builder! {
            @impl
            ($(#[$meta])*) $algorithm, $selector, $survivor,
            extras: [ ],
            override_build: false
        }
    };

    // -------- Single implementation (no duplication of setters/forwarders) --------
    (@impl
        ($(#[$meta:meta])*) $algorithm:ident, $selector:ty, $survivor:ty,
        extras: [ $( $ef:ident : $ety:ty ),* ],
        override_build: $ov:tt
    ) => {
        ::paste::paste! {
            // Public type alias
            $(#[$meta])*
            pub type $algorithm<S, Cross, Mut, F, G, DC> =
                $crate::algorithms::GeneticAlgorithm<
                    S, $selector, $survivor, Cross, Mut, F, G, DC
                >;

            // Specialized builder
            pub struct [<$algorithm Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: $crate::operators::SamplingOperator,
                $selector: $crate::operators::SelectionOperator<FDim = F::Dim>,
                $survivor: $crate::operators::SurvivalOperator<FDim = F::Dim>,
                Cross: $crate::operators::CrossoverOperator,
                Mut: $crate::operators::MutationOperator,
                F: $crate::evaluator::FitnessFn,
                G: $crate::evaluator::ConstraintsFn,
                DC: $crate::duplicates::PopulationCleaner,
            {
                inner: $crate::algorithms::AlgorithmBuilder<
                    S, $selector, $survivor, Cross, Mut, F, G, DC
                >,
                $( $ef: Option<$ety>, )*
            }

            // Default
            impl<S, Cross, Mut, F, G, DC> ::core::default::Default
                for [<$algorithm Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: $crate::operators::SamplingOperator,
                $selector: $crate::operators::SelectionOperator<FDim = F::Dim>,
                $survivor: $crate::operators::SurvivalOperator<FDim = F::Dim>,
                Cross: $crate::operators::CrossoverOperator,
                Mut: $crate::operators::MutationOperator,
                F: $crate::evaluator::FitnessFn,
                G: $crate::evaluator::ConstraintsFn,
                DC: $crate::duplicates::PopulationCleaner,
                $crate::algorithms::AlgorithmBuilder<
                    S, $selector, $survivor, Cross, Mut, F, G, DC
                >: ::core::default::Default,
            {
                fn default() -> Self {
                    let inner =
                        < $crate::algorithms::AlgorithmBuilder<
                            S, $selector, $survivor, Cross, Mut, F, G, DC
                        > as ::core::default::Default >::default()
                            .selector(<$selector as ::core::default::Default>::default())
                            .survivor(<$survivor as ::core::default::Default>::default());

                    Self {
                        inner,
                        $( $ef: None, )*
                    }
                }
            }

            // Setters (extras, if any) + forwarders (once)
            impl<S, Cross, Mut, F, G, DC> [<$algorithm Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: $crate::operators::SamplingOperator,
                $selector: $crate::operators::SelectionOperator<FDim = F::Dim>,
                $survivor: $crate::operators::SurvivalOperator<FDim = F::Dim>,
                Cross: $crate::operators::CrossoverOperator,
                Mut: $crate::operators::MutationOperator,
                F: $crate::evaluator::FitnessFn,
                G: $crate::evaluator::ConstraintsFn,
                DC: $crate::duplicates::PopulationCleaner,
            {
                $(
                    #[inline]
                    pub fn $ef(mut self, v: $ety) -> Self {
                        self.$ef = Some(v);
                        self
                    }
                )*

                // Forwarding methods to `inner`
                #[inline] pub fn sampler(mut self, v: S) -> Self { self.inner = self.inner.sampler(v); self }
                #[inline] pub fn crossover(mut self, v: Cross) -> Self { self.inner = self.inner.crossover(v); self }
                #[inline] pub fn mutation(mut self, v: Mut) -> Self { self.inner = self.inner.mutation(v); self }
                #[inline] pub fn selector(mut self, v: $selector) -> Self { self.inner = self.inner.selector(v); self }
                #[inline] pub fn survivor(mut self, v: $survivor) -> Self { self.inner = self.inner.survivor(v); self }
                #[inline] pub fn fitness_fn(mut self, v: F) -> Self { self.inner = self.inner.fitness_fn(v); self }
                #[inline] pub fn constraints_fn(mut self, v: G) -> Self { self.inner = self.inner.constraints_fn(v); self }
                #[inline] pub fn duplicates_cleaner(mut self, v: DC) -> Self { self.inner = self.inner.duplicates_cleaner(v); self }
                #[inline] pub fn num_vars(mut self, v: usize) -> Self { self.inner = self.inner.num_vars(v); self }
                #[inline] pub fn population_size(mut self, v: usize) -> Self { self.inner = self.inner.population_size(v); self }
                #[inline] pub fn num_offsprings(mut self, v: usize) -> Self { self.inner = self.inner.num_offsprings(v); self }
                #[inline] pub fn num_iterations(mut self, v: usize) -> Self { self.inner = self.inner.num_iterations(v); self }
                #[inline] pub fn mutation_rate(mut self, v: f64) -> Self { self.inner = self.inner.mutation_rate(v); self }
                #[inline] pub fn crossover_rate(mut self, v: f64) -> Self { self.inner = self.inner.crossover_rate(v); self }
                #[inline] pub fn keep_infeasible(mut self, v: bool) -> Self { self.inner = self.inner.keep_infeasible(v); self }
                #[inline] pub fn verbose(mut self, v: bool) -> Self { self.inner = self.inner.verbose(v); self }
                #[inline] pub fn seed(mut self, v: u64) -> Self { self.inner = self.inner.seed(v); self }
            }
        }

        // Conditionally add `build` depending on the override_build flag.
        $crate::create_algorithm_and_builder! {
            @maybe_build $algorithm, $selector, $survivor, override_build: $ov
        }
    };

    // Emit `build` when override_build_method = false
    (@maybe_build $algorithm:ident, $selector:ty, $survivor:ty, override_build: false) => {
        ::paste::paste! {
            impl<S, Cross, Mut, F, G, DC> [<$algorithm Builder>]<S, Cross, Mut, F, G, DC>
            where
                S: $crate::operators::SamplingOperator,
                $selector: $crate::operators::SelectionOperator<FDim = F::Dim>,
                $survivor: $crate::operators::SurvivalOperator<FDim = F::Dim>,
                Cross: $crate::operators::CrossoverOperator,
                Mut: $crate::operators::MutationOperator,
                F: $crate::evaluator::FitnessFn,
                G: $crate::evaluator::ConstraintsFn,
                DC: $crate::duplicates::PopulationCleaner,
            {
                pub fn build(self)
                    -> Result<$algorithm<S, Cross, Mut, F, G, DC>, $crate::algorithms::AlgorithmBuilderError>
                {
                    Ok(self.inner.build()?)
                }
            }
        }
    };

    // Do not emit `build` when override_build_method = true
    (@maybe_build $algorithm:ident, $selector:ty, $survivor:ty, override_build: true) => {};
}
