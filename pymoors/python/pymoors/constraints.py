from typing import Callable

import numpy as np

from pymoors.typing import TwoDArray


class Constraints:
    """
    Encapsulates custom constraint functions and optional bound constraints,
    producing a combined constraint evaluation when called.

    The `__call__` method returns a 2D array that concatenates:
      1. The output of a user‑supplied `constraints_fn`, if provided.
      2. Lower‑bound violation values (`lower_bound - genes`), if `lower_bound` is set.
      3. Upper‑bound violation values (`genes - upper_bound`), if `upper_bound` is set.

    This method will be delegated to the Rust side once this issue is fixed:
    https://github.com/andresliszt/moo-rs/issues/208


    Parameters
    ----------
    constraints_fn : Callable[[TwoDArray], TwoDArray], optional
        A user‑provided function that maps the gene matrix to a 2D array
        of custom constraint evaluations.
    lower_bound : float, optional
        A scalar lower bound applied element‑wise to the genes. Violations
        are computed as `lower_bound - genes`.
    upper_bound : float, optional
        A scalar upper bound applied element‑wise to the genes. Violations
        are computed as `genes - upper_bound`.

    Raises
    ------
    ValueError
        If none of `constraints_fn`, `lower_bound`, or `upper_bound` is provided.
    """

    def __init__(
        self,
        *,
        constraints_fn: Callable[[TwoDArray], TwoDArray] | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        if not any(
            [
                constraints_fn is not None,
                lower_bound is not None,
                upper_bound is not None,
            ]
        ):
            raise ValueError(
                "At least constraints_fn, lower_bound or upper_bound must be set"
            )

        self.constraints_fn = constraints_fn
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, genes: TwoDArray) -> TwoDArray:
        parts = []
        # any custom constraint‑function output
        if self.constraints_fn is not None:
            parts.append(self.constraints_fn(genes))
        # lower‑bound violations
        if self.lower_bound is not None:
            parts.append(self.lower_bound - genes)
        # upper‑bound violations
        if self.upper_bound is not None:
            parts.append(genes - self.upper_bound)
        # if only one part, return it directly as 2D
        if len(parts) == 1:
            if parts[0].ndim == 1:
                return parts[0].reshape(-1, 1)
            return parts[0]
        # otherwise concatenate horizontally into a single 2D array
        return np.concatenate(parts, axis=1)
