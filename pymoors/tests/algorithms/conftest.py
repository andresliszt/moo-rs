from typing import Callable

import pytest

from pymoors.schemas import Individual


@pytest.fixture
def compare_exact_front() -> Callable[[list[Individual], list[Individual]], None]:
    def compare(
        output_pareto_front: list[Individual], real_pareto_front: list[Individual]
    ):
        def to_comparable_set(individuals: list[Individual]) -> set:
            return {
                (
                    tuple(ind.genes.tolist()),
                    tuple(ind.fitness.tolist()),
                    tuple(
                        ind.constraints.tolist() if ind.constraints is not None else []
                    ),
                )
                for ind in individuals
            }

        output_set = to_comparable_set(output_pareto_front)
        real_set = to_comparable_set(real_pareto_front)

        assert output_set == real_set

    return compare
