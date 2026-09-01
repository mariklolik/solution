import numpy as np
import pandas as pd

from targeted_route import route_by_category


def test_route_by_category_restores_input_order():
    matches = pd.DataFrame({"id1": [10, 20, 30, 40], "id2": [11, 21, 31, 41]})
    categories = pd.Series(["base", "expert", "base", "expert"])

    result = route_by_category(
        matches,
        categories,
        {"expert"},
        lambda rows, _: rows["id1"].to_numpy() + 1,
        lambda rows, _: rows["id1"].to_numpy() + 100,
    )

    np.testing.assert_array_equal(result, [11, 120, 31, 140])


def test_route_by_category_handles_empty_input():
    matches = pd.DataFrame({"id1": [], "id2": []})
    categories = pd.Series([], dtype=str)

    result = route_by_category(matches, categories, {"expert"}, lambda *_: [], lambda *_: [])

    assert result.dtype == np.float32
    assert result.size == 0
