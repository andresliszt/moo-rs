use ndarray::{Array2, ArrayViewMut1};

/// A post-mutation operator that repairs infeasible or out-of-domain individuals in place.
///
/// Implement this trait to enforce problem-specific constraints (e.g., permutation validity,
/// variable bounds, or feasibility) after crossover and mutation have been applied.
///
/// The default [`operate`] implementation calls [`repair`] on every individual.
/// Override [`operate`] if you need selective or batch repair logic.
pub trait RepairOperator: std::fmt::Debug + Send + Sync {
    /// Repairs a single individual in place.
    fn repair(&self, individual: ArrayViewMut1<f64>);

    /// Applies repair to every individual in the population.
    fn operate(&self, population: &mut Array2<f64>) {
        for individual in population.outer_iter_mut() {
            self.repair(individual);
        }
    }
}

/// No-op repair — satisfies the `RepairOperator` bound at zero cost.
///
/// This is the default when no repair is needed. The compiler will eliminate
/// all calls to it.
#[derive(Debug, Clone, Default)]
pub struct NoRepair;

impl RepairOperator for NoRepair {
    #[inline]
    fn repair(&self, _individual: ArrayViewMut1<f64>) {}

    #[inline]
    fn operate(&self, _population: &mut Array2<f64>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// Clamps every gene to [0.0, 1.0] — used to test a real repair implementation.
    #[derive(Debug)]
    struct ClampRepair {
        min: f64,
        max: f64,
    }

    impl RepairOperator for ClampRepair {
        fn repair(&self, mut individual: ArrayViewMut1<f64>) {
            individual.mapv_inplace(|g| g.clamp(self.min, self.max));
        }
    }

    #[test]
    fn no_repair_leaves_individual_unchanged() {
        let original = array![1.5, -0.3, 99.0];
        let mut individual = original.clone();
        NoRepair.repair(individual.view_mut());
        assert_eq!(individual, original);
    }

    #[test]
    fn no_repair_leaves_population_unchanged() {
        let original = array![[1.5, -0.3], [99.0, 0.5]];
        let mut pop = original.clone();
        NoRepair.operate(&mut pop);
        assert_eq!(pop, original);
    }

    #[test]
    fn clamp_repair_fixes_out_of_bounds_individual() {
        let repair = ClampRepair { min: 0.0, max: 1.0 };
        let mut individual = array![-0.5, 0.5, 1.5];
        repair.repair(individual.view_mut());
        assert_eq!(individual, array![0.0, 0.5, 1.0]);
    }

    #[test]
    fn operate_default_applies_repair_to_all_rows() {
        let repair = ClampRepair { min: 0.0, max: 1.0 };
        let mut pop = array![[-1.0, 0.5], [0.3, 2.0]];
        repair.operate(&mut pop);
        assert_eq!(pop, array![[0.0, 0.5], [0.3, 1.0]]);
    }

    #[test]
    fn operate_on_empty_population_does_not_panic() {
        let repair = ClampRepair { min: 0.0, max: 1.0 };
        let mut pop = Array2::<f64>::zeros((0, 3));
        repair.operate(&mut pop);
        assert_eq!(pop.nrows(), 0);
    }
}
