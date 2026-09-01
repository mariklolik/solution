import numpy as np


def route_by_category(matches, categories, routed_categories, base_predict, routed_predict):
    routed = categories.isin(routed_categories).to_numpy()
    result = None
    for mask, predict in ((~routed, base_predict), (routed, routed_predict)):
        if not mask.any():
            continue
        values = np.asarray(predict(matches.loc[mask].reset_index(drop=True), categories.loc[mask].reset_index(drop=True)))
        if result is None:
            result = np.empty(len(matches), dtype=values.dtype)
        result[mask] = values
    return result if result is not None else np.empty(0, dtype=np.float32)
