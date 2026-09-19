import numpy as np
import pytest

from pymoors import (
    GaussianMutation,
    GeneticAlgorithmSOO,
    Nsga2,
    RandomSamplingFloat,
    SimulatedBinaryCrossover,
)
from pymoors.typing import AlgorithmContext, ControlSignal, TwoDArray


def moo_fitness_fn(genes: TwoDArray) -> TwoDArray:
    x = genes[:, 0]
    y = genes[:, 1]
    f1 = x**2 + y**2
    f2 = (x - 1) ** 2 + (y - 1) ** 2
    return np.column_stack((f1, f2))


def soo_fitness_fn(genes: TwoDArray) -> TwoDArray:
    return genes.sum(axis=1)


@pytest.fixture
def moo_kwargs():
    return {
        "sampler": RandomSamplingFloat(min=0.0, max=1.0),
        "crossover": SimulatedBinaryCrossover(distribution_index=15),
        "mutation": GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        "fitness_fn": moo_fitness_fn,
        "num_vars": 2,
        "population_size": 20,
        "num_offsprings": 20,
        "mutation_rate": 0.1,
        "crossover_rate": 0.9,
        "verbose": False,
    }


class RecordingController:
    """Controller that just records every call to `observe`."""

    def __init__(self):
        self.calls = []

    def observe(
        self,
        iteration,
        genes: TwoDArray,
        fitness: TwoDArray,
        constraints: TwoDArray,
        context: AlgorithmContext,
    ):
        self.calls.append(
            {
                "iteration": iteration,
                "genes_shape": genes.shape,
                "fitness_shape": fitness.shape,
                "constraints_shape": constraints.shape,
                "context": context,
            }
        )
        return None


class StoppingController:
    """Controller that requests an early stop once it reaches `stop_at`."""

    def __init__(self, stop_at: int):
        self.stop_at = stop_at
        self.calls = 0

    def observe(
        self,
        iteration: int,
        genes: TwoDArray,
        fitness: TwoDArray,
        constraints: TwoDArray,
        context: AlgorithmContext,
    ) -> ControlSignal:
        self.calls += 1
        return {"stop": iteration >= self.stop_at}


class RateAdjustingController:
    """Controller that overrides mutation/crossover rates every iteration."""

    def observe(
        self,
        iteration: int,
        genes: TwoDArray,
        fitness: TwoDArray,
        constraints: TwoDArray,
        context: AlgorithmContext,
    ) -> ControlSignal:
        return {"mutation_rate": 0.5, "crossover_rate": 0.5}


class InvalidController:
    """Missing the required `observe` method."""

    def not_observe(self):
        pass


def test_controller_observe_called_each_iteration(moo_kwargs):
    controller = RecordingController()
    num_iterations = 5
    algorithm = Nsga2(
        **moo_kwargs,
        num_iterations=num_iterations,
        controller=controller,
    )
    algorithm.run()

    assert len(controller.calls) == num_iterations
    assert [call["iteration"] for call in controller.calls] == list(
        range(num_iterations)
    )

    for call in controller.calls:
        assert call["genes_shape"] == (
            moo_kwargs["population_size"],
            moo_kwargs["num_vars"],
        )
        assert call["fitness_shape"][0] == moo_kwargs["population_size"]
        assert call["constraints_shape"][0] == moo_kwargs["population_size"]

        context = call["context"]
        assert context["num_vars"] == moo_kwargs["num_vars"]
        assert context["population_size"] == moo_kwargs["population_size"]
        assert context["num_offsprings"] == moo_kwargs["num_offsprings"]
        assert context["num_iterations"] == num_iterations
        assert context["current_iteration"] == call["iteration"]


def test_controller_can_stop_algorithm_early(moo_kwargs):
    stop_at = 2
    controller = StoppingController(stop_at=stop_at)
    algorithm = Nsga2(
        **moo_kwargs,
        num_iterations=10,
        controller=controller,
    )
    algorithm.run()

    # observe is called for iterations 0, 1, 2 and then the run breaks
    assert controller.calls == stop_at + 1


def test_controller_can_adjust_rates(moo_kwargs):
    algorithm = Nsga2(
        **moo_kwargs,
        num_iterations=5,
        controller=RateAdjustingController(),
    )
    algorithm.run()

    assert len(algorithm.population) == moo_kwargs["population_size"]


def test_controller_missing_observe_method_raises(moo_kwargs):
    with pytest.raises(
        TypeError,
        match="Custom controller class must define an 'observe' method",
    ):
        Nsga2(
            **moo_kwargs,
            num_iterations=5,
            controller=InvalidController(),  # type: ignore
        )


def test_controller_with_soo_algorithm():
    controller = RecordingController()
    num_iterations = 4
    population_size = 10
    algorithm = GeneticAlgorithmSOO(
        sampler=RandomSamplingFloat(min=0.0, max=10.0),
        mutation=GaussianMutation(gene_mutation_rate=0.5, sigma=0.01),
        crossover=SimulatedBinaryCrossover(distribution_index=15.0),
        fitness_fn=soo_fitness_fn,
        num_vars=2,
        population_size=population_size,
        num_offsprings=population_size,
        num_iterations=num_iterations,
        mutation_rate=0.1,
        crossover_rate=0.9,
        controller=controller,
    )
    algorithm.run()

    assert len(controller.calls) == num_iterations
    for call in controller.calls:
        # SOO fitness is a 1D array, unlike MOO's 2D array
        assert len(call["fitness_shape"]) == 1
        assert call["fitness_shape"][0] == population_size
