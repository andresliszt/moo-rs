import numpy as np

from pymoors import ExactDuplicatesCleaner, CloseDuplicatesCleaner

# 1729 is the most beautiful number, is the smallest number that can be written as the sum of two cubes
# in the two different ways: 10**3 + 9**3 = 12**3 + 1**3 = 1729


def test_duplicates():
    population = np.array(
        [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1729, 1729, 1729, 1279],
            [1729, 1729, 1729, 1279],
        ],
        dtype=float,
    )

    exact_duplicates_cleaner = ExactDuplicatesCleaner()
    close_duplicates_cleaner = CloseDuplicatesCleaner(epsilon=1e-2)

    expected = np.array(
        [[1, 2, 3, 4], [1729, 1729, 1729, 1279]],
        dtype=float,
    )

    np.testing.assert_array_equal(
        exact_duplicates_cleaner.remove_duplicates(population=population),  # type: ignore
        expected,
    )
    np.testing.assert_array_equal(
        close_duplicates_cleaner.remove_duplicates(population=population),  # type: ignore
        expected,
    )


def test_duplicates_with_references():
    population = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1729, 1729, 1729, 1279],
        ],
        dtype=float,
    )

    reference = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        dtype=float,
    )

    exact_duplicates_cleaner = ExactDuplicatesCleaner()
    close_duplicates_cleaner = CloseDuplicatesCleaner(epsilon=1e-2)

    expected = np.array(
        [[1729, 1729, 1729, 1279]],
        dtype=float,
    )

    np.testing.assert_array_equal(
        exact_duplicates_cleaner.remove_duplicates(
            population=population, reference=reference
        ),
        expected,
    )
    np.testing.assert_array_equal(
        close_duplicates_cleaner.remove_duplicates(
            population=population, reference=reference
        ),
        expected,
    )
