#[macro_export]
macro_rules! define_algorithm_and_builder {
    // === API: with override_build_method  ===
    (
        $(#[$meta:meta])*
        $algorithm:ident, $selector:ty, $survivor:ty
        $(, selection_args = [ $( $larg:ident : $lty:ty ),* $(,)? ])?
        $(, survival_args  = [ $( $sarg:ident : $sty:ty ),* $(,)? ])?
        $(, shared_survival_args = [ $( $ss:ident : $ssty:ty ),* $(,)? ])?
        , override_build_method = $ov:tt
        $(,)?
    ) => {
        define_algorithm_and_builder! {
            @impl
            ($(#[$meta])*) $algorithm, $selector, $survivor,
            selection: [ $( $( $larg : $lty ),* )? ],
            survival:  [ $( $( $sarg : $sty ),* )? ],
            shared:    [ $( $( $ss   : $ssty),* )? ],
            override_build: $ov
        }
    };

    // === API: w.o override_build_method (default = false) ===
    (
        $(#[$meta:meta])*
        $algorithm:ident, $selector:ty, $survivor:ty
        $(, selection_args = [ $( $larg:ident : $lty:ty ),* $(,)? ])?
        $(, survival_args  = [ $( $sarg:ident : $sty:ty ),* $(,)? ])?
        $(, shared_survival_args = [ $( $ss:ident : $ssty:ty ),* $(,)? ])?
        $(,)? // trailing comma opcional
    ) => {
        define_algorithm_and_builder!(
            $(#[$meta])*
            $algorithm, $selector, $survivor,
            selection_args = [ $( $( $larg : $lty ),* )? ],
            survival_args  = [ $( $( $sarg : $sty ),* )? ],
            shared_survival_args = [ $( $( $ss : $ssty ),* )? ],
            override_build_method = false
        );
    };

    // ============================ Impl block =============================
    (@impl
        ($(#[$meta:meta])*) $algorithm:ident, $selector:ty, $survivor:ty,
        selection: [ $( $larg:ident : $lty:ty ),* ],
        survival:  [ $( $sarg:ident : $sty:ty ),* ],
        shared:    [ $( $ss:ident   : $ssty:ty ),* ],
        override_build: $ov:tt
    ) => {
        ::paste::paste! {
            $(#[$meta])*
            pub type $algorithm<S, Cross, Mut, F, G, DC> =
                $crate::algorithms::GeneticAlgorithm<
                    S, $selector, $survivor, Cross, Mut, F, G, DC
                >;

            // -------- Builder -------------------------------------------------
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

                $( $larg: ::core::option::Option<$lty>, )*
                $( $sarg: ::core::option::Option<$sty>, )*
            }

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
                    Self {
                        inner: ::core::default::Default::default(),
                        $( $larg: ::core::option::Option::None, )*
                        $( $sarg: ::core::option::Option::None, )*
                    }
                }
            }

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
                // === Public setters (selection/survival) ====================
                $(
                    #[inline]
                    pub fn $larg(mut self, v: $lty) -> Self {
                        self.$larg = ::core::option::Option::Some(v);
                        self
                    }
                )*
                $(
                    #[inline]
                    pub fn $sarg(mut self, v: $sty) -> Self {
                        self.$sarg = ::core::option::Option::Some(v);
                        self
                    }
                )*

                // === Inner Forwards ============================
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

                // === Build =====================================================
                pub fn build(mut self) -> ::core::result::Result<
                    $algorithm<S, Cross, Mut, F, G, DC>,
                    $crate::algorithms::AlgorithmBuilderError
                > {
                    if $ov {
                        // We do not rebuild selector/survivor: use what is already set
                        return self.inner.build();
                    }

                    // --- Selector::new( selection_args... ) ---
                    $(
                        let $larg = self.$larg.ok_or(
                            $crate::algorithms::AlgorithmBuilderError::UninitializedField(
                                ::core::concat!(::core::module_path!(), "::", ::core::stringify!($larg))
                            )
                        )?;
                    )*
                    let selector_val = define_algorithm_and_builder!(@call_selector $selector ; ( $( $larg ),* ));
                    self.inner = self.inner.selector(selector_val);

                    // --- Survivor::new( shared..., survival_args... ) ---
                    $(
                        let $ss = self.inner.[<$ss>].ok_or(
                            $crate::algorithms::AlgorithmBuilderError::UninitializedField(
                                ::core::concat!(::core::module_path!(), "::", ::core::stringify!($ss))
                            )
                        )?;
                    )*
                    $(
                        let $sarg = self.$sarg.ok_or(
                            $crate::algorithms::AlgorithmBuilderError::UninitializedField(
                                ::core::concat!(::core::module_path!(), "::", ::core::stringify!($sarg))
                            )
                        )?;
                    )*

                    let survivor_val = define_algorithm_and_builder!(
                        @call_survivor $survivor ; ( $( $sarg ),* ) ( $( $ss ),* )
                    );

                    self.inner = self.inner.survivor(survivor_val);

                    self.inner.build()
                }
            }
        }
    };

    (@call_selector $ty:ty ; () ) => { < $ty >::new() };
    (@call_selector $ty:ty ; ( $( $a:ident ),+ ) ) => { < $ty >::new( $( $a ),+ ) };

    // 0/0
    (@call_survivor $ty:ty ; () () ) => {
        < $ty >::new()
    };

    // A/0  (A = survival_args)
    (@call_survivor $ty:ty ; ( $( $a:ident ),+ ) () ) => {
        < $ty >::new( $( $a ),+ )
    };

    // 0/B  (B = shared_survival_args)
    (@call_survivor $ty:ty ; () ( $( $b:ident ),+ ) ) => {
        < $ty >::new( $( $b ),+ )
    };

    // A/B  (survival first, then shared)
    (@call_survivor $ty:ty ; ( $( $a:ident ),+ ) ( $( $b:ident ),+ ) ) => {
        < $ty >::new( $( $a ),+ , $( $b ),+ )
    };

}
