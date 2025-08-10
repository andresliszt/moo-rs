from typing import Callable, TypeAlias
import numpy as np

from pymoors.typing import TwoDArray, OneDArray

ConstraintSpec: TypeAlias = (
    Callable[[TwoDArray], OneDArray]
    | Callable[[TwoDArray], TwoDArray]
    | list[Callable[[TwoDArray], OneDArray] | Callable[[TwoDArray], TwoDArray]]
    | None
)


class Constraints:
    """
    Encapsulates equality (`eq`) and inequality (`ineq`) constraint functions plus optional bound constraints.

    When called with `genes` (shape: (n, d)), returns a 2D array formed by horizontally
    concatenating, in order:
      1) Outputs from `eq` (single callable or list). Each callable maps `genes -> (n,) or (n, k)`.
         Each equality residual h(x) is converted to an inequality via ε-technique:
         `|h(x)| - epsilon`, yielding values ≤ 0 when within tolerance.
         1D outputs are reshaped to (n, 1).
      2) Outputs from `ineq` (single callable or list). Each callable maps `genes -> (n,) or (n, k)`.
         1D outputs are reshaped to (n, 1).
      3) Lower-bound violations: (lower_bound - genes), if provided (shape: (n, d)).
      4) Upper-bound violations: (genes - upper_bound), if provided (shape: (n, d)).

    The resulting matrix has shape (n, m), where:
      m = (cols from ε-adapted `eq`) + (cols from `ineq`)
          + d * I[lower_bound is not None] + d * I[upper_bound is not None]

    Parameters
    ----------
    eq : Callable[[TwoDArray], OneDArray | TwoDArray]
         | list[Callable[[TwoDArray], OneDArray | TwoDArray]]
         | None
        Equality constraint residuals h(x). Each output is converted to `|h(x)| - epsilon`.
    ineq : Callable[[TwoDArray], OneDArray | TwoDArray]
           | list[Callable[[TwoDArray], OneDArray | TwoDArray]]
           | None
        Inequality constraints g(x) ≤ 0.
    lower_bound : float, optional
        Scalar lower bound applied element-wise; violations are (lower_bound - genes).
    upper_bound : float, optional
        Scalar upper bound applied element-wise; violations are (genes - upper_bound).
    epsilon : float, optional (default: 1e-6)
        Tolerance for equality constraints. Must be non-negative.

    Returns
    -------
    TwoDArray
        Constraint evaluation matrix of shape (n, m), as described above.

    Raises
    ------
    ValueError
        If none of `eq`, `ineq`, `lower_bound`, or `upper_bound` is provided;
        if any callable returns an array whose first dimension differs from `n`;
        if a callable returns an array with ndim > 2; or if `epsilon < 0`.
    """

    def __init__(
        self,
        *,
        eq: ConstraintSpec = None,
        ineq: ConstraintSpec = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        epsilon: float = 1e-6,
    ):
        if epsilon < 0:
            raise ValueError("`epsilon` must be non-negative.")

        def _to_callable_list(
            spec: ConstraintSpec,
        ) -> list[Callable[[TwoDArray], OneDArray] | Callable[[TwoDArray], TwoDArray]]:
            if spec is None:
                return []
            if isinstance(spec, list):
                return spec
            return [spec]

        # Normalize to lists so Pyright knows these are callables (not Sequence)
        self.eq: list[
            Callable[[TwoDArray], OneDArray] | Callable[[TwoDArray], TwoDArray]
        ] = _to_callable_list(eq)
        self.ineq: list[
            Callable[[TwoDArray], OneDArray] | Callable[[TwoDArray], TwoDArray]
        ] = _to_callable_list(ineq)

        if not any(
            [
                len(self.eq) > 0,
                len(self.ineq) > 0,
                lower_bound is not None,
                upper_bound is not None,
            ]
        ):
            raise ValueError(
                "At least one of `eq`, `ineq`, `lower_bound`, or `upper_bound` must be provided."
            )

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.epsilon = float(epsilon)

    def _normalize_output(self, arr: np.ndarray, n_rows: int) -> np.ndarray:
        """Ensure constraint output is 2D with shape (n_rows, k)."""
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Constraint function returned shape {arr.shape}, expected ({n_rows},) for 1D output."
                )
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            if arr.shape[0] != n_rows:
                raise ValueError(
                    f"Constraint function returned shape {arr.shape}, expected first dimension {n_rows}."
                )
            return arr
        raise ValueError(
            f"Constraint function returned array with ndim={arr.ndim}; only 1D or 2D supported."
        )

    def __call__(self, genes: TwoDArray) -> TwoDArray:
        n_rows = genes.shape[0]
        parts: list[np.ndarray] = []

        # 1) Equality residuals -> ε-inequalities: |h(x)| - epsilon
        for fn in self.eq:
            raw = fn(genes)
            norm = self._normalize_output(raw, n_rows)
            parts.append(np.abs(norm) - self.epsilon)

        # 2) Inequalities as-is
        for fn in self.ineq:
            out = fn(genes)
            parts.append(self._normalize_output(out, n_rows))

        # 3) Bounds
        if self.lower_bound is not None:
            parts.append(self.lower_bound - genes)
        if self.upper_bound is not None:
            parts.append(genes - self.upper_bound)

        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=1)
