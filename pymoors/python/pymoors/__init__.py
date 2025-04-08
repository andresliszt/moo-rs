from typing import overload, Literal
import textwrap

from ._pymoors import (
    Nsga2,
    Nsga3,
    AgeMoea,
    Rnsga2,
    Revea,
    BitFlipMutation,
    SwapMutation,
    GaussianMutation,
    DisplacementMutation,
    ScrambleMutation,
    RandomSamplingBinary,
    RandomSamplingFloat,
    RandomSamplingInt,
    PermutationSampling,
    SinglePointBinaryCrossover,
    OrderCrossover,
    UniformBinaryCrossover,
    SimulatedBinaryCrossover,
    ExponentialCrossover,
    ExactDuplicatesCleaner,
    CloseDuplicatesCleaner,
    NoFeasibleIndividualsError,
    InvalidParameterError,
    DanAndDenisReferencePoints,
)


def _available_operators(
    op_list, include_docs: bool = False
) -> list[str] | dict[str, str]:
    # List only the classes that are crossover operators
    if not include_docs:
        return [cls.__name__ for cls in op_list]
    operators = {}
    for cls in op_list:
        # Extract the docstring, and if present, use the first non-empty line as a description.
        doc = cls.__doc__
        if doc:
            # Split into lines and take the first non-empty line
            clean_doc = textwrap.dedent(doc).strip()
            description = next(
                (line.strip() for line in clean_doc.splitlines() if line.strip()),
                "No description available.",
            )
        else:
            description = "No description available."
        operators[cls.__name__] = description
    return operators


@overload
def available_operators(
    operator_type: Literal["crossover", "sampling", "mutation", "duplicates"],
    include_docs: Literal[True],
) -> dict[str, str]: ...


@overload
def available_operators(
    operator_type: Literal["crossover", "sampling", "mutation", "duplicates"],
    include_docs: Literal[False],
) -> list[str]: ...


def available_operators(
    operator_type: Literal["crossover", "sampling", "mutation", "duplicates"],
    include_docs: bool = False,
) -> list[str] | dict[str, str]:
    if operator_type == "crossover":
        return _available_operators(
            op_list=[
                SinglePointBinaryCrossover,
                OrderCrossover,
                UniformBinaryCrossover,
                ExponentialCrossover,
                SimulatedBinaryCrossover,
            ],
            include_docs=include_docs,
        )
    if operator_type == "mutation":
        return _available_operators(
            op_list=[
                BitFlipMutation,
                SwapMutation,
                GaussianMutation,
                ScrambleMutation,
                DisplacementMutation,
            ],
            include_docs=include_docs,
        )
    if operator_type == "sampling":
        return _available_operators(
            op_list=[
                RandomSamplingBinary,
                RandomSamplingFloat,
                RandomSamplingInt,
                PermutationSampling,
            ],
            include_docs=include_docs,
        )
    if operator_type == "duplicates":
        return _available_operators(
            op_list=[
                ExactDuplicatesCleaner,
                CloseDuplicatesCleaner,
            ],
            include_docs=include_docs,
        )
    raise ValueError(
        "operator_type must be one of: crossover, mutation, sampling or duplicates"
    )


__all__ = (
    "AgeMoea",
    "Nsga2",
    "Nsga3",
    "Rnsga2",
    "Revea",
    "BitFlipMutation",
    "SwapMutation",
    "GaussianMutation",
    "DisplacementMutation",
    "ScrambleMutation",
    "RandomSamplingBinary",
    "RandomSamplingFloat",
    "PermutationSampling",
    "RandomSamplingInt",
    "SinglePointBinaryCrossover",
    "OrderCrossover",
    "UniformBinaryCrossover",
    "ExponentialCrossover",
    "SimulatedBinaryCrossover",
    "ExactDuplicatesCleaner",
    "CloseDuplicatesCleaner",
    "NoFeasibleIndividualsError",
    "InvalidParameterError",
    "available_operators",
    "DanAndDenisReferencePoints",
)
