from typing import Annotated, Callable, TypeAlias, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

OneDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=1"]
TwoDArray: TypeAlias = Annotated[npt.NDArray[DType], "ndim=2"]

FitnessPopulationCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]
ConstraintsPopulationCallable: TypeAlias = Callable[[TwoDArray], TwoDArray]
