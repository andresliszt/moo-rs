import pytest
from pymoors import (
    RandomSamplingFloat,
    SimulatedBinaryCrossover,
    GaussianMutation,
    Nsga2,
    InvalidParameterError,
)


@pytest.fixture
def valid_algorithm_params():
    """Returns valid parameters for Nsga2."""
    return {
        "sampler": RandomSamplingFloat(min=0.0, max=1.0),
        "crossover": SimulatedBinaryCrossover(distribution_index=2),
        "mutation": GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        "fitness_fn": lambda genes: genes,  # Mock function for fitness evaluation
        "n_vars": 10,
        "population_size": 100,
        "n_offsprings": 50,
        "n_iterations": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.9,
        "keep_infeasible": False,
        "verbose": False,
        "lower_bound": 0,
        "upper_bound": 1,
        "seed": 42,
    }


# ✅ **Test that a valid setup does not raise errors**
def test_valid_algorithm_creation(valid_algorithm_params):
    """Ensure that valid parameters do not raise exceptions."""
    try:
        Nsga2(**valid_algorithm_params)
    except Exception as e:
        pytest.fail(f"Valid parameters raised an exception: {e}")


# ❌ **Test that mutation_rate out of range raises an error**
@pytest.mark.parametrize("invalid_mutation_rate", [-0.1, 1.5])
def test_invalid_mutation_rate(valid_algorithm_params, invalid_mutation_rate):
    """Mutation rate must be between 0 and 1."""
    valid_algorithm_params["mutation_rate"] = invalid_mutation_rate
    with pytest.raises(
        InvalidParameterError, match="Mutation rate must be between 0 and 1"
    ):
        Nsga2(**valid_algorithm_params)


# ❌ **Test that crossover_rate out of range raises an error**
@pytest.mark.parametrize("invalid_crossover_rate", [-0.5, 2.0])
def test_invalid_crossover_rate(valid_algorithm_params, invalid_crossover_rate):
    """Crossover rate must be between 0 and 1."""
    valid_algorithm_params["crossover_rate"] = invalid_crossover_rate
    with pytest.raises(
        InvalidParameterError, match="Crossover rate must be between 0 and 1"
    ):
        Nsga2(**valid_algorithm_params)


# ❌ **Test that n_vars cannot be zero**
def test_invalid_n_vars(valid_algorithm_params):
    """Number of variables must be greater than 0."""
    valid_algorithm_params["n_vars"] = 0
    with pytest.raises(
        InvalidParameterError, match="Number of variables must be greater than 0"
    ):
        Nsga2(**valid_algorithm_params)


# ❌ **Test that population size must be greater than 0**
def test_invalid_population_size(valid_algorithm_params):
    """Population size must be greater than 0."""
    valid_algorithm_params["population_size"] = 0
    with pytest.raises(
        InvalidParameterError, match="Population size must be greater than 0"
    ):
        Nsga2(**valid_algorithm_params)


# ❌ **Test that number of offsprings must be greater than 0**
def test_invalid_n_offsprings(valid_algorithm_params):
    """Number of offsprings must be greater than 0."""
    valid_algorithm_params["n_offsprings"] = 0
    with pytest.raises(
        InvalidParameterError, match="Number of offsprings must be greater than 0"
    ):
        Nsga2(**valid_algorithm_params)


# ❌ **Test that number of iterations must be greater than 0**
def test_invalid_n_iterations(valid_algorithm_params):
    """Number of iterations must be greater than 0."""
    valid_algorithm_params["n_iterations"] = 0
    with pytest.raises(
        InvalidParameterError, match="Number of iterations must be greater than 0"
    ):
        Nsga2(**valid_algorithm_params)


# ❌ **Test that lower_bound must be less than upper_bound**
@pytest.mark.parametrize("lower, upper", [(1.0, 1.0), (2.0, 1.0)])
def test_invalid_bounds(valid_algorithm_params, lower, upper):
    """Lower bound must be less than upper bound."""
    valid_algorithm_params["lower_bound"] = lower
    valid_algorithm_params["upper_bound"] = upper
    with pytest.raises(
        InvalidParameterError, match="Lower bound .* must be less than upper bound .*"
    ):
        Nsga2(**valid_algorithm_params)
