from typing import Annotated, Callable, Protocol, TypeAlias, TypedDict, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)

OneDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=1"]
TwoDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=2"]

FitnessCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]
ConstraintsCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]


class AlgorithmContext(TypedDict):
    """Runtime state passed to a controller's `observe` method on every iteration."""

    num_vars: int
    population_size: int
    num_offsprings: int
    num_iterations: int
    current_iteration: int
    upper_bound: float | None
    lower_bound: float | None


class ControlSignal(TypedDict, total=False):
    """Optional adjustments a controller may return from `observe` to steer the algorithm."""

    mutation_rate: float
    crossover_rate: float
    stop: bool


class CrossoverProtocol(Protocol):
    def operate(
        self, parents_a: TwoDArray, parents_b: TwoDArray, seed: int | None
    ) -> TwoDArray: ...


class MutationProtocol(Protocol):
    def operate(self, population: TwoDArray, seed: int | None) -> TwoDArray: ...


class CrossoverProtocolNoSeed(Protocol):
    def operate(self, parents_a: TwoDArray, parents_b: TwoDArray) -> TwoDArray: ...


class MutationProtocolNoSeed(Protocol):
    def operate(self, population: TwoDArray) -> TwoDArray: ...


class SamplingProtocol(Protocol):
    def operate(
        self, population_size: int, num_vars: int, seed: int | None
    ) -> TwoDArray: ...


class SamplingProtocolNoArgs(Protocol):
    def operate(self) -> TwoDArray: ...


class ControllerProtocol(Protocol):
    def observe(
        self,
        iteration: int,
        genes: TwoDArray,
        fitness: TwoDArray,
        constraints: TwoDArray,
        context: AlgorithmContext,
    ) -> ControlSignal | None: ...


CrossoverLike = CrossoverProtocol | CrossoverProtocolNoSeed
MutationLike = MutationProtocol | MutationProtocolNoSeed
SamplingLike = SamplingProtocol | SamplingProtocolNoArgs
ControllerLike = ControllerProtocol
