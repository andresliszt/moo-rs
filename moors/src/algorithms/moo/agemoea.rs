use crate::{
    define_algorithm_and_builder, selection::moo::AgeMoeaRankCrowdingSelection,
    survival::moo::AgeMoeaSurvival,
};

define_algorithm_and_builder!(
    /// AGE-MOEA algorithm wrapper.
    ///
    /// This struct is a thin facade over [`GeneticAlgorithm`] preset with
    /// the AGE-MOEA survival and selection strategy.
    ///
    /// * **Selection:** [`RankAndScoringSelection`]
    /// * **Survival:**  [`AgeMoeaSurvival`] (elitist, adaptive geometry estimation)
    ///
    /// Construct it with [`AgeMoeaBuilder`](crate::algorithms::AgeMoeaBuilder).
    /// After building, call [`run`](GeneticAlgorithm::run)
    /// and then [`population`](GeneticAlgorithm::population) to retrieve the
    /// final non-dominated set.
    ///
    /// For algorithmic details, see:
    /// Annibale Panichella (2019),
    /// "An Adaptive Evolutionary Algorithm based on Non-Euclidean Geometry for Many-objective Optimization",
    /// in *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19)*,
    /// pp. 595–603, July 2019.
    /// DOI: 10.1145/3321707.3321839
    AgeMoea,
    AgeMoeaRankCrowdingSelection,
    AgeMoeaSurvival
);
