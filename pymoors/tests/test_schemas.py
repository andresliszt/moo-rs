import numpy as np
import pytest
from pymoors.schemas import Individual, Population


def test_individual_is_best():
    genes = np.array([0.1, 0.2, 0.3])
    fitness = np.array([0.9, 0.8, 0.7])
    constraints = np.array([0.0, -0.1, -0.2])

    individual = Individual(
        genes=genes, fitness=fitness, rank=0, constraints=constraints
    )
    assert individual.is_best

    individual.rank = 1
    assert not individual.is_best


def test_individual_is_feasible():
    genes = np.array([0.1, 0.2, 0.3])
    fitness = np.array([0.9, 0.8, 0.7])
    constraints = np.array([0.0, -0.1, -0.2])

    # Feasible constraints
    individual = Individual(
        genes=genes, fitness=fitness, rank=0, constraints=constraints
    )
    assert individual.is_feasible

    # Infeasible constraints
    individual.constraints = np.array([0.0, 0.1, -0.2])
    assert not individual.is_feasible

    # No constraints
    individual.constraints = None
    assert individual.is_feasible


def test_population_length():
    genes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fitness = np.array([[0.5, 0.4], [0.7, 0.6], [0.8, 0.9]])
    rank = np.array([0, 1, 2])
    constraints = np.array([[0.0, -0.1], [0.0, 0.1], [-0.2, 0.0]])

    pop = Population(genes, fitness, rank, constraints)
    assert len(pop) == 3


def test_population_best():
    genes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fitness = np.array(
        [
            [0.5, 0.4],
            [0.4, 0.5],
            [0.7, 0.6],
        ]
    )
    rank = np.array([0, 0, 1])

    pop = Population(genes, fitness, rank, None)
    assert len(pop.best) == 2
    assert len(pop.best_as_population) == 2
    assert isinstance(pop.best_as_population, Population)


def test_population_best_no_rank():
    # W.O rank every individual is considered best (for now)
    genes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fitness = np.array(
        [
            [0.5, 0.4],
            [0.4, 0.5],
            [0.7, 0.6],
        ]
    )

    pop = Population(genes, fitness, None, None)
    assert len(pop.best) == 3
    assert len(pop.best_as_population) == 3
    assert isinstance(pop.best_as_population, Population)


def test_population_getitem():
    genes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fitness = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
    rank = np.array([0, 1, 2])
    constraints = np.array([[0.0, -0.1], [0.0, 0.1], [-0.2, 0.0]])

    pop = Population(genes, fitness, rank, constraints)

    # Test single indexing
    individual = pop[0]
    assert np.array_equal(individual.genes, genes[0])
    assert np.array_equal(individual.fitness, fitness[0])
    assert individual.rank == rank[0]
    assert np.array_equal(individual.constraints, constraints[0])  # type: ignore

    # Test slicing
    individuals = pop[1:3]
    assert len(individuals) == 2
    assert np.array_equal(individuals[0].genes, genes[1])
    assert np.array_equal(individuals[1].genes, genes[2])

    # Test Incorrect get item
    with pytest.raises(
        TypeError, match="indices must be integers or slices, not <class 'list'>"
    ):
        _ = pop[[1, 2, 3]]  # type: ignore


def test_population_raises_value_error_for_length_mismatch():
    genes = np.array([[0.1, 0.2], [0.3, 0.4]])
    fitness = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
    rank = np.array([0, 1])
    constraints = np.array([[0.0, -0.1], [0.0, 0.1]])

    with pytest.raises(
        ValueError, match="genes and fitness arrays must have the same lenght"
    ):
        Population(genes, fitness, rank, constraints)


def test_population_raises_value_error_for_constraints_mismatch():
    genes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fitness = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
    rank = np.array([0, 1, 2])
    constraints = np.array([[0.0, -0.1]])

    with pytest.raises(
        ValueError, match="constraints must have the same length as genes"
    ):
        Population(genes, fitness, rank, constraints)


def test_population_raises_value_error_for_rank_mismatch():
    genes = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    fitness = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]])
    rank = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError, match="rank must have the same length as genes"):
        Population(genes, fitness, rank, None)
