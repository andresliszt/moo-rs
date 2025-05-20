from typing import Annotated, Callable, Protocol, TypeAlias, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

OneDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=1"]
TwoDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=2"]

FitnessPopulationCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]
ConstraintsPopulationCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]


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


CrossoverFn: TypeAlias = Callable[[TwoDArray, TwoDArray], TwoDArray]
MutationFn: TypeAlias = Callable[[TwoDArray], TwoDArray]


CrossoverLike = CrossoverProtocol | CrossoverProtocolNoSeed | CrossoverFn
MutationLike = MutationProtocol | MutationProtocolNoSeed | MutationFn
