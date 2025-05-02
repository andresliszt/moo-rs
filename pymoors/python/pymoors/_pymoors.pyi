from typing import Optional, TypedDict

from typing_extensions import Unpack

from pymoors.schemas import Population
from pymoors.typing import (
    FitnessPopulationCallable,
    ConstraintsPopulationCallable,
    TwoDArray,
)

# pylint: disable=W0622, W0231

class SamplingOperator:
    """
    Base class for sampling operators used to initialize or generate new individuals in the population.

    This abstract class defines the interface for different sampling strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the SamplingOperator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    def sample(
        self, population_size: int, n_vars: int, seed: Optional[int]
    ) -> TwoDArray: ...

class MutationOperator:
    """
    Base class for mutation operators used to introduce variations in individuals.

    This abstract class defines the interface for different mutation strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the MutationOperator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    def muatation(self, population: TwoDArray, seed: Optional[int]) -> TwoDArray: ...

class CrossoverOperator:
    """
    Base class for crossover operators used to combine two parent individuals to produce offspring.

    This abstract class defines the interface for different crossover strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the CrossoverOperator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

    def crossover(
        self, parents_a: TwoDArray, parents_b: TwoDArray, seed: Optional[int]
    ) -> TwoDArray: ...

class RandomSamplingFloat(SamplingOperator):
    """
    Sampling operator for floating-point variables using uniform random distribution.

    Generates random float values within a specified range.

    Args:
        min (float): The minimum value for sampling.
        max (float): The maximum value for sampling.
    """

    def __init__(self, min: float, max: float) -> None: ...

class RandomSamplingInt(SamplingOperator):
    """
    Sampling operator for integer variables using uniform random distribution.

    Generates random integer values within a specified range.

    Args:
        min (int): The minimum integer value for sampling.
        max (int): The maximum integer value for sampling.
    """

    def __init__(self, min: int, max: int) -> None: ...

class RandomSamplingBinary(SamplingOperator):
    """
    Sampling operator for binary variables.

    Generates random binary values (0 or 1).
    """

    def __init__(self) -> None: ...

class PermutationSampling(SamplingOperator):
    """
    Sampling operator for permutation-based variables.

    Generates random permutations of a given set of elements.
    """

    def __init__(self) -> None: ...

class BitFlipMutation(MutationOperator):
    """
    Mutation operator that flips bits in a binary individual with a specified mutation rate.

    Each bit has a probability equal to `gene_mutation_rate` to be flipped.

    Args:
        gene_mutation_rate (float): The probability of flipping each bit.
    """

    def __init__(self, gene_mutation_rate: float) -> None: ...

class SwapMutation(MutationOperator):
    """
    Mutation operator that swaps two genes in a permutation-based individual.

    This operator is useful for permutation-based representations to maintain valid permutations.
    """

    def __init__(self) -> None: ...

class GaussianMutation(MutationOperator):
    """
    Mutation operator that adds Gaussian noise to float variables.

    Each gene is perturbed by a Gaussian-distributed random value.

    Args:
        gene_mutation_rate (float): The probability of mutating each gene.
        sigma (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, gene_mutation_rate: float, sigma: float) -> None: ...

class DisplacementMutation(MutationOperator):
    """Operator that extracts a segment from an individual and reinserts it at a new random position."""
    def __init__(self) -> None: ...

class ScrambleMutation(MutationOperator):
    """Operator that selects a random segment from an individual and scrambles it by randomly reordering the genes within the segment."""
    def __init__(self) -> None: ...

class OrderCrossover(CrossoverOperator):
    """
    Crossover operator for permutation-based individuals using Order Crossover (OX).

    Preserves the relative order of genes from the parents in the offspring.
    """

    def __init__(self) -> None: ...

class SinglePointBinaryCrossover(CrossoverOperator):
    """
    Single-point crossover operator for binary-encoded individuals.

    A single crossover point is selected, and the binary strings are exchanged beyond that point.
    """

    def __init__(self) -> None: ...

class UniformBinaryCrossover(CrossoverOperator):
    """
    Uniform binary crossover operator for genetic algorithms.

    This operator performs uniform crossover on binary representations.
    Given two parent solutions, each gene of the offspring is independently
    selected at random from one of the parents with equal probability.
    This approach facilitates a balanced mix of the genetic material from both
    parents, enhancing the diversity of the resulting population.
    """
    def __init__(self) -> None: ...

class ExponentialCrossover(CrossoverOperator):
    """
    Crossover operator that combines parent genes based on an exponential distribution.

    The `exponential_crossover_rate` controls the influence of each parent.

    Args:
        exponential_crossover_rate (float): The rate parameter for the exponential distribution.
    """

    def __init__(self, exponential_crossover_rate: float) -> None: ...

class SimulatedBinaryCrossover(CrossoverOperator):
    """
    Simulated Binary Crossover (SBX) operator for real-coded genetic algorithms.

    SBX is a widely used crossover mechanism for continuous variables, inspired
    by the behavior of single-point crossover in binary-coded GAs. Instead of
    slicing bit strings, SBX generates offspring by interpolating (and potentially
    extrapolating) between two parent solutions (p1 and p2).

    The key parameter `distribution_index` (often called "eta") controls how far
    the offspring can deviate from the parents. A higher distribution index results
    in offspring closer to the parents (exploitation), whereas a lower value
    produces offspring that can be further away (exploration).

    In each crossover event, SBX computes a factor `beta_q`, based on a random
    number in [0,1) and the distribution index `eta`. This factor dictates where
    each child solution lies relative to the parent solutions. If the parent genes
    (p1 and p2) differ minimally, no crossover is performed (i.e., the children
    inherit the parents' values directly).

    Reference:
        - Deb, Kalyanmoy, and R. B. Agrawal. "Simulated binary crossover for
          continuous search space." Complex Systems 9.2 (1995): 115-148.

    """
    def __init__(self, distribution_index: float): ...

class DuplicatesCleaner:
    """
    Base class for cleaning duplicate individuals in the population.

    This abstract class defines the interface for different duplicate cleaning strategies.
    """

    def remove_duplicates(
        self, population: TwoDArray, reference: Optional[TwoDArray]
    ) -> TwoDArray:
        """Removes duplicates from population."""

class ExactDuplicatesCleaner(DuplicatesCleaner):
    """
    Cleaner that removes exact duplicate individuals from the population.

    Ensures all individuals in the population are unique.
    """

    def __init__(self) -> None: ...

class CloseDuplicatesCleaner(DuplicatesCleaner):
    """
    Cleaner that removes individuals that are close to each other based on a specified epsilon.

    Two individuals are considered duplicates if their distance is less than `epsilon`.

    Args:
        epsilon (float): The distance threshold to consider individuals as duplicates.
    """

    def __init__(self, epsilon: float) -> None: ...

# Reference Points

class StructuredReferencePoints:
    def generate(self) -> TwoDArray: ...

class DanAndDenisReferencePoints(StructuredReferencePoints):
    """Generates reference points for multi-objective optimization using the Das–Dennis procedure.
    This class implements the reference point generation method as described by Das and Dennis,
    which is widely used in algorithms such as NSGA-III. The procedure partitions the objective
    space by generating a uniformly distributed set of points on the simplex.

    The process is as follows:
    1. Determine a parameter H (number of divisions) such that the total number of generated
        reference points (given by the binomial coefficient C(H + m - 1, m - 1)) is at least the
        desired number of reference points.
    2. Generate all combinations of nonnegative integers (h1, h2, ..., hm) that sum to H.
    3. Normalize each combination by dividing each component by H so that the resulting point lies
        on the m-dimensional simplex (i.e., the components sum to 1).

    Parameters:
    n_reference_points (int): The desired number of reference points (controls the granularity).
    n_objectives (int): The number of objectives in the optimization problem.

    Reference:
    Das, I. and Dennis, B. (1998). "Normal Boundary Intersection: A New Method for Generating
    the Pareto Surface in Multiple Objective Optimization." Evolutionary Computation, 8(3),
    377–400.
    """
    def __init__(self, n_reference_points: int, n_objectives: int) -> None: ...

# Algorithms

class _MooAlgorithmKwargs(TypedDict, total=False):
    """
    It exists for Multi-Objective Optimization (MOO) algorithms kwargs.

    Provides common functionalities and interfaces for MOO algorithms like NSGA-II and NSGA-III.

    Args:
        sampler (SamplingOperator): Operator to sample initial population.
        crossover (CrossoverOperator): Operator to perform crossover.
        mutation (MutationOperator): Operator to perform mutation.
        fitness_fn (FitnessPopuAgeMoealationCallable): Function to evaluate the fitness of the population.
        n_vars (int): Number of variables in the optimization problem.
        population_size (int): Population size.
        num_offsprings (int): Number of offsprings generated in each generation.
        num_iterations (int): Number of generations to run the algorithm.
        mutation_rate (float): Probability of mutation.
        crossover_rate (float): Probability of crossover.
        keep_infeasible (bool, optional): Whether to keep infeasible solutions. Defaults to False.
        verbose (bool, optional): Whether to print detailed information during the run. Defaults to True.
        duplicates_cleaner (Optional[DuplicatesCleaner], optional): Cleaner to remove duplicates. Defaults to None.
        constraints_fn (Optional[ConstraintsPopulationCallable], optional): Function to handle constraints. Defaults to None.
        lower_bound (Optional[float], optional): Optional lower bound for each gene. Defaults to None.
        upper_bound (Optional[float], optional): Optional upper bound for each gene. Defaults to None.
        seed (Optional[seed], optional): Optional seed to control experiments. Defaults to None.
    """

    sampler: SamplingOperator
    crossover: CrossoverOperator
    mutation: MutationOperator
    fitness_fn: FitnessPopulationCallable
    n_vars: int
    population_size: int
    num_offsprings: int
    num_iterations: int
    mutation_rate: float
    crossover_rate: float
    keep_infeasible: bool
    verbose: bool
    duplicates_cleaner: Optional[DuplicatesCleaner]
    constraints_fn: Optional[ConstraintsPopulationCallable]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    seed: Optional[int]

class Nsga2:
    """
    Implementation of the NSGA-II (Non-dominated Sorting Genetic Algorithm II).

    NSGA-II is a popular multi-objective evolutionary algorithm known for its fast non-dominated sorting approach
    and crowding distance mechanism to maintain diversity in the population.

    Reference:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm:
        NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
    """

    def __init__(self, **kwargs: Unpack[_MooAlgorithmKwargs]): ...
    @property
    def population(self) -> Population:
        """
        Get the current population.

        Returns:
            Population: The current population of individuals.
        """

    def run(self) -> None: ...

class _Nsga3Kwargs(_MooAlgorithmKwargs, total=False):
    reference_points: TwoDArray | StructuredReferencePoints

class Nsga3:
    """
    Implementation of the NSGA-III (Non-dominated Sorting Genetic Algorithm III).

    NSGA-III extends NSGA-II to handle many-objective optimization problems by introducing reference points
    for better diversity maintenance in higher dimensions.

    Reference:
        Deb, K., Jain, H., & Thiele, L. (2014). NSGA-III: A Many-Objective Genetic Algorithm Using Reference-Points
        for Selection. IEEE Transactions on Evolutionary Computation, 18(4), 577-601.
    """

    def __init__(self, **kwargs: Unpack[_Nsga3Kwargs]) -> None: ...
    @property
    def population(self) -> Population:
        """
        Get the current population.

        Returns:
            Population: The current population of individuals.
        """

    def run(self) -> None: ...

class _RNsg2Kwargs(_MooAlgorithmKwargs, total=False):
    reference_points: TwoDArray | StructuredReferencePoints
    epsilon: float

class Rnsga2:
    """
    Implementation of Rnsga2 (Reference-based NSGA-II).

    Rnsga2 is a variant of NSGA-II that incorporates reference points into the selection process
    to enhance diversity in many-objective optimization problems. By integrating a reference
    point–based ranking into the survival operator, Rnsga2 aims to obtain a well-distributed
    approximation of the Pareto front even in high-dimensional objective spaces.

    References:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
            A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
            IEEE Transactions on Evolutionary Computation, 6(2), 182–197.

        Deb, K., & Jain, H. (2014).
            An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
            Nondominated Sorting Approach, Part I: Solving Problems with Box Constraints.
            IEEE Transactions on Evolutionary Computation, 18(4), 577–601.

    """

    def __init__(self, **kwargs: Unpack[_RNsg2Kwargs]) -> None: ...
    @property
    def population(self) -> Population:
        """
        Returns the current population of individuals.

        Returns:
            Population: The current population.
        """

    def run(self) -> None: ...

class AgeMoea:
    """
    Adaptive Evolutionary Algorithm for Many-Objective Optimization based on Non-Euclidean Geometry.

    The algorithm employs adaptive strategies based on non-Euclidean geometry to effectively maintain diversity
    and drive convergence in many-objective optimization problems.

    References:
      Annibale Panichella, (2019).
        An adaptive evolutionary algorithm based on non-euclidean geometry for many-objective optimization.
        Evolutionary Computation Conference, GECCO, New York, NY, USA.

    """
    def __init__(self, **kwargs: Unpack[_MooAlgorithmKwargs]): ...
    @property
    def population(self) -> Population:
        """
        Get the current population.

        Returns:
            Population: The current population of individuals.
        """

    def run(self) -> None: ...

class _ReveaKwargs(_MooAlgorithmKwargs, total=False):
    reference_points: TwoDArray | StructuredReferencePoints
    alpha: float
    frequency: float

class Revea:
    def __init__(self, **kwargs: Unpack[_ReveaKwargs]) -> None: ...
    @property
    def population(self) -> Population:
        """
        Returns the current population of individuals.

        Returns:
            Population: The current population.
        """

    def run(self) -> None: ...

# Custom Errors

class NoFeasibleIndividualsError(BaseException):
    """Raise this error when no feasible individuals are found"""

class InvalidParameterError(BaseException):
    """Raise this error when an invalid parameter is provided"""
