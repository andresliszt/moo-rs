import pytest


from pymoors import available_operators


@pytest.mark.parametrize(
    "operator_type", ["crossover", "mutation", "sampling", "duplicates", "unknown"]
)
def test_available_operators(operator_type):
    if operator_type == "unknown":
        with pytest.raises(ValueError):
            available_operators(operator_type=operator_type)  # type: ignore
    else:
        assert isinstance(
            available_operators(operator_type=operator_type, include_docs=False), list
        )
        assert isinstance(
            available_operators(operator_type=operator_type, include_docs=True), dict
        )
