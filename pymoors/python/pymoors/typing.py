from typing import Annotated, Callable, Protocol, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)

OneDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=1"]
TwoDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=2"]

FitnessCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]
ConstraintsCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]


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


CrossoverLike = CrossoverProtocol | CrossoverProtocolNoSeed
MutationLike = MutationProtocol | MutationProtocolNoSeed
SamplingLike = SamplingProtocol | SamplingProtocolNoArgs
