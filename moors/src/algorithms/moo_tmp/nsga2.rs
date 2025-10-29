use crate::{
    create_algorithm_and_builder,
    operators::{
        selection::moo::RankAndScoringSelection, survival::moo::Nsga2RankCrowdingSurvival,
    },
};

create_algorithm_and_builder!(Nsga2, RankAndScoringSelection, Nsga2RankCrowdingSurvival);
