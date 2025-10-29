mod builder;
pub(crate) mod helpers;
mod macros;
mod moo;
mod moo_tmp;
mod soo;

pub use builder::{AlgorithmBuilder, AlgorithmBuilderError, GeneticAlgorithm};
pub use moo::agemoea::{AgeMoea, AgeMoeaBuilder};
pub use moo::ibea::{Ibea, IbeaBuilder};
pub use moo::nsga2::{Nsga2, Nsga2Builder};
pub use moo::nsga3::{Nsga3, Nsga3Builder};
pub use moo::revea::{Revea, ReveaBuilder};
pub use moo::rnsga2::{Rnsga2, Rnsga2Builder};
pub use moo::spea2::{Spea2, Spea2Builder};
pub use moo_tmp::nsga2::Nsga2Builder as Nsga2BuilderTmp;
pub use moo_tmp::nsga3::Nsga3Builder as Nsga3BuilderTmp;

pub use helpers::{AlgorithmError, InitializationError};
