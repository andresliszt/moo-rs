use ndarray::Array2;

use crate::operators::survival::moo::reference_points::StructuredReferencePoints;

#[derive(Debug, Clone)]
pub struct DanAndDenisReferencePoints {
    n_reference_points: usize,
    num_objectives: usize,
}

pub struct NormalBoundaryDivisions {
    pub outer_divisions: usize,
    pub inner_divisions: usize,
}

impl NormalBoundaryDivisions {
    pub fn new(outer_divisions: usize, inner_divisions: usize) -> Self {
        Self {
            outer_divisions,
            inner_divisions,
        }
    }

    /// Pick the *default* boundary divisions for a given number of objectives
    pub fn for_num_objectives(num_obj: usize) -> Self {
        match num_obj {
            1 => Self::new(100, 0),
            2 => Self::new(99, 0),
            3 => Self::new(12, 0),
            4 => Self::new(8, 0),
            5 => Self::new(6, 0),
            6 => Self::new(4, 1),
            7..=10 => Self::new(3, 2),
            _ => Self::new(2, 1),
        }
    }
}

impl DanAndDenisReferencePoints {
    pub fn new(n_reference_points: usize, num_objectives: usize) -> Self {
        Self {
            n_reference_points,
            num_objectives,
        }
    }

    pub fn from_divisions(divisions: usize, num_objectives: usize) -> Self {
        // Compute exactly how many points that will yield:
        //   C(divisions + m - 1, m - 1)
        let n_points = binomial_coefficient(divisions + num_objectives - 1, num_objectives - 1);
        DanAndDenisReferencePoints::new(n_points, num_objectives)
    }
}

impl StructuredReferencePoints for DanAndDenisReferencePoints {
    /// Generates all Das-Dennis reference points given a population size and number of objectives.
    ///
    /// The procedure is:
    /// 1. Estimate H using `choose_h(population_size, m)`.
    /// 2. Generate all combinations of nonnegative integers (h₁, h₂, …, hₘ) that satisfy:
    ///    h₁ + h₂ + ... + hₘ = H.
    /// 3. Normalize each combination by dividing each component by H to get a point on the simplex.
    ///
    /// The function returns an Array2<f64> where each row is a reference point.
    fn generate(&self) -> Array2<f64> {
        // Step 1: Estimate H using the population size and number of objectives.
        let h = choose_h(self.n_reference_points, self.num_objectives);

        // Step 2: Generate all combinations (h₁, h₂, …, hₘ) such that h₁ + h₂ + ... + hₘ = H.
        let mut points: Vec<Vec<usize>> = Vec::new();
        let mut current: Vec<usize> = Vec::with_capacity(self.num_objectives);
        generate_combinations(self.num_objectives, h, 0, &mut current, &mut points);

        // Step 3: Normalize each combination by dividing by H and store in an Array2.
        let num_points = points.len();
        let mut arr = Array2::<f64>::zeros((num_points, self.num_objectives));
        for (i, combination) in points.iter().enumerate() {
            for j in 0..self.num_objectives {
                arr[[i, j]] = combination[j] as f64 / h as f64;
            }
        }
        arr
    }
}

/// Returns the smallest value of H such that the number of Das-Dennis reference points
/// (computed as binom(H + m - 1, m - 1)) is greater than or equal to `n_reference_points`.
fn choose_h(n_reference_points: usize, num_objectives: usize) -> usize {
    let mut h = 1;
    loop {
        let n_points = binomial_coefficient(h + num_objectives - 1, num_objectives - 1);
        if n_points >= n_reference_points {
            return h;
        }
        h += 1;
    }
}

/// Computes the binomial coefficient "n choose k".
fn binomial_coefficient(n: usize, k: usize) -> usize {
    let mut result = 1;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Recursively generates all combinations of nonnegative integers of length `m` that sum to `sum`.
///
/// - `num_objectives`: total number of objectives
/// - `sum`: the remaining sum to distribute among the components
/// - `index`: current index being filled
/// - `current`: holds the current combination under construction
/// - `points`: collects all generated combinations
fn generate_combinations(
    num_objectives: usize,
    sum: usize,
    index: usize,
    current: &mut Vec<usize>,
    points: &mut Vec<Vec<usize>>,
) {
    if index == num_objectives - 1 {
        // For the last component, assign the remaining sum.
        current.push(sum);
        points.push(current.clone());
        current.pop();
        return;
    }
    // Distribute values from 0 to `sum` for the current component.
    for x in 0..=sum {
        current.push(x);
        generate_combinations(num_objectives, sum - x, index + 1, current, points);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use crate::StructuredReferencePoints;
    use crate::survival::moo::reference_points::dan_and_dennis::{
        DanAndDenisReferencePoints, NormalBoundaryDivisions,
    };
    use ndarray::Axis;

    #[test]
    fn test_dan_and_dennis() {
        let num_obj = 4;
        let divs = NormalBoundaryDivisions::for_num_objectives(num_obj);
        assert_eq!(divs.outer_divisions, 8);
        assert_eq!(divs.inner_divisions, 0);

        let mut ref_dirs =
            DanAndDenisReferencePoints::from_divisions(divs.outer_divisions, num_obj).generate();
        // inner layer at H = inner_divisions
        if divs.inner_divisions > 0 {
            let inner = DanAndDenisReferencePoints::from_divisions(divs.inner_divisions, num_obj)
                .generate();
            ref_dirs
                .append(Axis(0), inner.view())
                .expect("shapes must be compatible");
        }

        println!("Ref dirs: {:#?}", ref_dirs);
    }
}
