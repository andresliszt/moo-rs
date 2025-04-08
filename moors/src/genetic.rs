use ndarray::{Array1, Array2, ArrayViewMut1, Axis, concatenate};

/// Represents an individual in the population.
/// Each `IndividualGenes` is an `Array1<f64>`.
pub type IndividualGenes = Array1<f64>;
pub type IndividualGenesMut<'a> = ArrayViewMut1<'a, f64>;

/// Represents an individual with genes, fitness, constraints (if any),
/// rank, and an optional survival score.
pub struct Individual {
    pub genes: IndividualGenes,
    pub fitness: Array1<f64>,
    pub constraints: Option<Array1<f64>>,
    pub rank: Option<usize>,
    pub survival_score: Option<f64>,
}

impl Individual {
    pub fn new(
        genes: IndividualGenes,
        fitness: Array1<f64>,
        constraints: Option<Array1<f64>>,
        rank: Option<usize>,
        survival_score: Option<f64>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            survival_score,
        }
    }

    pub fn is_feasible(&self) -> bool {
        match &self.constraints {
            None => true,
            Some(c) => c.iter().sum::<f64>() <= 0.0,
        }
    }
}

/// Type aliases to work with populations.
pub type PopulationGenes = Array2<f64>;
pub type PopulationFitness = Array2<f64>;
pub type PopulationConstraints = Array2<f64>;
/// Type aliases for functions
pub type FitnessFunc = fn(&PopulationGenes) -> PopulationFitness;
pub type ConstraintsFn = fn(&PopulationGenes) -> PopulationConstraints;

/// The `Population` struct contains genes, fitness, constraints (if any),
/// rank (optional), and optionally a survival score vector.
#[derive(Debug)]
pub struct Population {
    pub genes: PopulationGenes,
    pub fitness: PopulationFitness,
    pub constraints: Option<PopulationConstraints>,
    pub rank: Option<Array1<usize>>,
    pub survival_score: Option<Array1<f64>>,
}

impl Clone for Population {
    fn clone(&self) -> Self {
        Self {
            genes: self.genes.clone(),
            fitness: self.fitness.clone(),
            constraints: self.constraints.clone(),
            rank: self.rank.clone(),
            survival_score: self.survival_score.clone(),
        }
    }
}

impl Population {
    /// Creates a new `Population` instance with the given genes, fitness, constraints, and rank.
    /// The `survival_score` field is set to `None` by default.
    pub fn new(
        genes: PopulationGenes,
        fitness: PopulationFitness,
        constraints: Option<PopulationConstraints>,
        rank: Option<Array1<usize>>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            survival_score: None, // Initialized to None by default.
        }
    }

    /// Retrieves an `Individual` from the population by index.
    pub fn get(&self, idx: usize) -> Individual {
        let constraints = self.constraints.as_ref().map(|c| c.row(idx).to_owned());
        let survival_score = self.survival_score.as_ref().map(|ss| ss[idx]);
        let rank = self.rank.as_ref().map(|r| r[idx]);
        Individual::new(
            self.genes.row(idx).to_owned(),
            self.fitness.row(idx).to_owned(),
            constraints,
            rank,
            survival_score,
        )
    }

    /// Returns a new `Population` containing only the individuals at the specified indices.
    pub fn selected(&self, indices: &[usize]) -> Population {
        let genes = self.genes.select(Axis(0), indices);
        let fitness = self.fitness.select(Axis(0), indices);
        let rank = self.rank.as_ref().map(|r| r.select(Axis(0), indices));
        let survival_score = self
            .survival_score
            .as_ref()
            .map(|ss| ss.select(Axis(0), indices));
        let constraints = self
            .constraints
            .as_ref()
            .map(|c| c.select(Axis(0), indices));

        let mut selected_population = Population::new(genes, fitness, constraints, rank);
        if let Some(score) = survival_score {
            selected_population
                .set_survival_score(score)
                .expect("Failed to set survival score");
        }
        selected_population
    }

    /// Returns the number of individuals in the population.
    pub fn len(&self) -> usize {
        self.genes.nrows()
    }

    /// Returns a new `Population` containing only the individuals with rank = 0.
    /// If no ranking information is available, the entire population is returned.
    pub fn best(&self) -> Population {
        if let Some(ranks) = &self.rank {
            let indices: Vec<usize> = ranks
                .iter()
                .enumerate()
                .filter_map(|(i, &r)| if r == 0 { Some(i) } else { None })
                .collect();
            self.selected(&indices)
        } else {
            // If rank is not set, return the entire population.
            self.clone()
        }
    }

    /// Updates the population's `survival_score` field.
    ///
    /// This method validates that the provided `diversity` vector has the same number of elements
    /// as individuals in the population. If not, it returns an error.
    pub fn set_survival_score(&mut self, score: Array1<f64>) -> Result<(), String> {
        if score.len() != self.len() {
            return Err(format!(
                "The diversity vector has length {} but the population contains {} individuals.",
                score.len(),
                self.len()
            ));
        }
        self.survival_score = Some(score);
        Ok(())
    }

    /// Merges two populations into one.
    pub fn merge(population1: &Population, population2: &Population) -> Population {
        // Concatenate genes (assumed to be an Array2).
        let merged_genes = concatenate(
            Axis(0),
            &[population1.genes.view(), population2.genes.view()],
        )
        .expect("Failed to merge genes");

        // Concatenate fitness (assumed to be an Array2).
        let merged_fitness = concatenate(
            Axis(0),
            &[population1.fitness.view(), population2.fitness.view()],
        )
        .expect("Failed to merge fitness");

        // Merge rank: both must be Some or both must be None.
        let merged_rank = match (&population1.rank, &population2.rank) {
            (Some(r1), Some(r2)) => {
                Some(concatenate(Axis(0), &[r1.view(), r2.view()]).expect("Failed to merge rank"))
            }
            (None, None) => None,
            _ => panic!("Mismatched population rank: one is set and the other is None"),
        };

        // Merge constraints: both must be Some or both must be None.
        let merged_constraints = match (&population1.constraints, &population2.constraints) {
            (Some(c1), Some(c2)) => Some(
                concatenate(Axis(0), &[c1.view(), c2.view()]).expect("Failed to merge constraints"),
            ),
            (None, None) => None,
            _ => panic!("Mismatched population constraints: one is set and the other is None"),
        };

        // Merge survival_score: both must be Some or both must be None.
        let merged_survival_score = match (&population1.survival_score, &population2.survival_score)
        {
            (Some(s1), Some(s2)) => Some(
                concatenate(Axis(0), &[s1.view(), s2.view()])
                    .expect("Failed to merge survival scores"),
            ),
            (None, None) => None,
            _ => panic!("Mismatched population survival scores: one is set and the other is None"),
        };

        let mut merged_population = Population::new(
            merged_genes,
            merged_fitness,
            merged_constraints,
            merged_rank,
        );

        if let Some(score) = merged_survival_score {
            merged_population
                .set_survival_score(score)
                .expect("Failed to set survival score");
        }
        merged_population
    }
}

/// Type alias for a vector of `Population` representing multiple fronts.
pub type Fronts = Vec<Population>;

/// An extension trait for `Fronts` that adds a `.to_population()` method
/// which flattens multiple fronts into a single `Population`.
pub trait FrontsExt {
    fn to_population(self) -> Population;
}

impl FrontsExt for Vec<Population> {
    fn to_population(self) -> Population {
        self.into_iter()
            .reduce(|pop1, pop2| Population::merge(&pop1, &pop2))
            .expect("Error when merging population vector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_individual_is_feasible() {
        // Individual with no constraints should be feasible.
        let ind1 = Individual::new(array![1.0, 2.0], array![0.5, 1.0], None, Some(0), None);
        assert!(
            ind1.is_feasible(),
            "Individual with no constraints should be feasible"
        );

        // Individual with constraints summing to <= 0 is feasible.
        let ind2 = Individual::new(
            array![1.0, 2.0],
            array![0.5, 1.0],
            Some(array![-1.0, 0.0]),
            Some(0),
            None,
        );
        assert!(
            ind2.is_feasible(),
            "Constraints sum -1.0 should be feasible"
        );

        // Individual with constraints summing to > 0 is not feasible.
        let ind3 = Individual::new(
            array![1.0, 2.0],
            array![0.5, 1.0],
            Some(array![1.0, 0.1]),
            Some(0),
            None,
        );
        assert!(
            !ind3.is_feasible(),
            "Constraints sum 1.1 should not be feasible"
        );
    }

    #[test]
    fn test_population_new_get_selected_len() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        // Using a rank array here.
        let rank = Some(array![0, 1]);
        let pop = Population::new(genes.clone(), fitness.clone(), None, rank);

        // Test len()
        assert_eq!(pop.len(), 2, "Population should have 2 individuals");

        // Test get()
        let ind0 = pop.get(0);
        assert_eq!(ind0.genes, genes.row(0).to_owned());
        assert_eq!(ind0.fitness, fitness.row(0).to_owned());
        assert_eq!(ind0.rank, Some(0));

        // Test selected()
        let selected = pop.selected(&[1]);
        assert_eq!(
            selected.len(),
            1,
            "Selected population should have 1 individual"
        );
        let ind_selected = selected.get(0);
        assert_eq!(ind_selected.genes, array![3.0, 4.0]);
        assert_eq!(ind_selected.fitness, array![1.5, 2.0]);
        assert_eq!(ind_selected.rank, Some(1));
    }

    #[test]
    fn test_population_best_with_rank() {
        // Create a population with three individuals and varying ranks.
        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]];
        // First and third individuals have rank 0, second has rank 1.
        let rank = Some(array![0, 1, 0]);
        let pop = Population::new(genes, fitness, None, rank);
        let best = pop.best();
        // Expect best population to contain only individuals with rank 0.
        assert_eq!(best.len(), 2, "Best population should have 2 individuals");
        for i in 0..best.len() {
            let ind = best.get(i);
            assert_eq!(
                ind.rank,
                Some(0),
                "All individuals in best population should have rank 0"
            );
        }
    }

    #[test]
    fn test_population_best_without_rank() {
        // Create a population without rank information.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let pop = Population::new(genes.clone(), fitness.clone(), None, None);
        // Since there is no rank, best() should return the whole population.
        let best = pop.best();
        assert_eq!(
            best.len(),
            pop.len(),
            "Best population should equal the original population when rank is None"
        );
    }

    #[test]
    fn test_set_survival_score() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let rank = Some(array![0, 1]);
        let mut pop = Population::new(genes, fitness, None, rank);
        // Set a survival score vector with correct length.
        let score = array![0.1, 0.2];
        assert!(pop.set_survival_score(score.clone()).is_ok());
        assert_eq!(pop.survival_score.unwrap(), score);
    }

    #[test]
    fn test_set_survival_score_err() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let rank = Some(array![0, 1]);
        let mut pop = Population::new(genes, fitness, None, rank);

        // Setting a survival score vector with incorrect length should error.
        let wrong_score = array![0.1];
        assert!(pop.set_survival_score(wrong_score).is_err());
    }

    #[test]
    fn test_population_merge() {
        // Create two populations with rank information.
        let genes1 = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness1 = array![[0.5, 1.0], [1.5, 2.0]];
        let rank1 = Some(array![0, 0]);
        let pop1 = Population::new(genes1, fitness1, None, rank1);

        let genes2 = array![[5.0, 6.0], [7.0, 8.0]];
        let fitness2 = array![[2.5, 3.0], [3.5, 4.0]];
        let rank2 = Some(array![1, 1]);
        let pop2 = Population::new(genes2, fitness2, None, rank2);

        let merged = Population::merge(&pop1, &pop2);
        assert_eq!(
            merged.len(),
            4,
            "Merged population should have 4 individuals"
        );

        let expected_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(merged.genes, expected_genes, "Merged genes do not match");

        let expected_fitness = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]];
        assert_eq!(
            merged.fitness, expected_fitness,
            "Merged fitness does not match"
        );

        let expected_rank = Some(array![0, 0, 1, 1]);
        assert_eq!(merged.rank, expected_rank, "Merged rank does not match");
    }

    #[test]
    fn test_fronts_ext_to_population() {
        // Create two fronts.
        let genes1 = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness1 = array![[0.5, 1.0], [1.5, 2.0]];
        let rank1 = Some(array![0, 0]);
        let pop1 = Population::new(genes1, fitness1, None, rank1);

        let genes2 = array![[5.0, 6.0], [7.0, 8.0]];
        let fitness2 = array![[2.5, 3.0], [3.5, 4.0]];
        let rank2 = Some(array![1, 1]);
        let pop2 = Population::new(genes2, fitness2, None, rank2);

        let fronts: Vec<Population> = vec![pop1.clone(), pop2.clone()];
        let merged = fronts.to_population();

        assert_eq!(
            merged.len(),
            4,
            "Flattened population should have 4 individuals"
        );

        let expected_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(merged.genes, expected_genes, "Flattened genes do not match");
    }

    #[test]
    #[should_panic(
        expected = "Mismatched population constraints: one is set and the other is None"
    )]
    fn test_population_merge_mismatched_constraints() {
        // Crear dos poblaciones con constraints incompatibles: una con Some y otra sin.
        let genes1 = array![[1.0, 2.0]];
        let fitness1 = array![[0.5, 1.0]];
        let constraints1 = Some(array![[-1.0, 0.0]]);
        let pop1 = Population::new(genes1, fitness1, constraints1, None);

        let genes2 = array![[3.0, 4.0]];
        let fitness2 = array![[1.5, 2.0]];
        // pop2 sin constraints.
        let pop2 = Population::new(genes2, fitness2, None, None);

        Population::merge(&pop1, &pop2);
    }

    #[test]
    #[should_panic(
        expected = "Mismatched population survival scores: one is set and the other is None"
    )]
    fn test_population_merge_mismatched_survival_score() {
        // Crear dos poblaciones con survival_score incompatibles: una con Some y otra sin.
        let genes1 = array![[1.0, 2.0]];
        let fitness1 = array![[0.5, 1.0]];
        let mut pop1 = Population::new(genes1, fitness1, None, None);
        // Asignar survival_score a pop1.
        let score1 = array![0.1];
        pop1.set_survival_score(score1).unwrap();

        let genes2 = array![[3.0, 4.0]];
        let fitness2 = array![[1.5, 2.0]];
        // pop2 sin survival_score.
        let pop2 = Population::new(genes2, fitness2, None, None);

        Population::merge(&pop1, &pop2);
    }
}
