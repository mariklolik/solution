import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from structured_features import serialize_item
from targeted_route import route_by_category


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "reranker"
ADAPTED_MODEL_PATH = ROOT / "models" / "adapted_reranker"
AMBIGUOUS_MODEL_PATH = ROOT / "models" / "ambiguous_reranker"
WEAK_MODEL_PATH = ROOT / "models" / "weak_reranker"
VERIFIED_CATEGORIES = {
    "Мебель",
    "Обувь",
}
AMBIGUOUS_CATEGORIES = {
    "Галантерея и аксессуары",
    "Одежда",
    "Ювелирные изделия",
}
PRIMARY_MAX_LENGTH = 384
SECONDARY_MAX_LENGTH = 256
MAX_CHARS = 3600
PRIMARY_BATCH_SIZE = 384
SECONDARY_BATCH_SIZE = 1024
WEAK_RERANKER_WEIGHT = 0.275
PRIMARY_REVERSE_FRACTION = 0.5
SECONDARY_REVERSE_FRACTION = 0.0
PRIMARY_REVERSE_FRACTIONS = None
WEAK_RERANKER_WEIGHTS = None


def relevant_items(path, required_ids):
    remaining = set(map(int, required_ids))
    frames = []
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=262144, columns=["id", "name", "attributes", "category"], use_threads=True):
        frame = batch.to_pandas()
        selected = frame[frame["id"].isin(remaining)]
        if len(selected):
            frames.append(selected)
            remaining.difference_update(map(int, selected["id"]))
        if not remaining:
            break
    if remaining:
        raise ValueError(f"Missing {len(remaining)} item ids")
    return pd.concat(frames, ignore_index=True)


def item_texts(items):
    return {
        int(item_id): serialize_item(name, attributes, category, MAX_CHARS)
        for item_id, name, attributes, category in items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None)
    }


def predict_direction(model, tokenizer, left_ids, right_ids, texts, order, max_length, batch_size):
    predictions = np.empty(len(left_ids), dtype=np.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            left = [texts[int(left_ids[index])] for index in indices]
            right = [texts[int(right_ids[index])] for index in indices]
            tokens = tokenizer(
                left,
                right,
                padding=True,
                truncation="longest_first",
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to("cuda", non_blocking=True) for key, value in tokens.items()}
            logits = model(**tokens).logits.float().reshape(-1)
            predictions[indices] = torch.sigmoid(logits).cpu().numpy()
            if offset and offset % (batch_size * 50) == 0:
                print(json.dumps({"reranker_pairs": offset, "seconds": time.perf_counter() - started}), flush=True)
    return predictions


def reranker_predictions(matches, texts, model_path, max_length, batch_size, categories, reverse_fraction, category_reverse_fractions=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda").eval()
    left_ids = matches["id1"].to_numpy(dtype=np.int64)
    right_ids = matches["id2"].to_numpy(dtype=np.int64)
    lengths = np.fromiter((len(texts[int(left)]) + len(texts[int(right)]) for left, right in zip(left_ids, right_ids)), dtype=np.int32, count=len(matches))
    order = np.argsort(lengths, kind="stable")
    forward = predict_direction(model, tokenizer, left_ids, right_ids, texts, order, max_length, batch_size)
    prediction = forward
    reverse_fractions = categories.map(category_reverse_fractions).fillna(reverse_fraction).to_numpy() if category_reverse_fractions else reverse_fraction
    if np.any(reverse_fractions > 0):
        reverse_mask = category_ranks(forward, categories) > 1 - reverse_fractions
        reverse_order = order[reverse_mask[order]]
        reverse = predict_direction(model, tokenizer, right_ids, left_ids, texts, reverse_order, max_length, batch_size)
        prediction = forward.copy()
        prediction[reverse_mask] = (forward[reverse_mask] + reverse[reverse_mask]) / 2
    del model
    torch.cuda.empty_cache()
    return prediction


def category_ranks(prediction, categories):
    frame = pd.DataFrame({"prediction": prediction, "category": categories})
    return frame.groupby("category")["prediction"].rank(method="average", pct=True).to_numpy()


def blend(reranker, weak_reranker, categories, category_weights=None):
    reranker_ranks = category_ranks(reranker, categories)
    if weak_reranker is None:
        return reranker_ranks
    weights = categories.map(category_weights).fillna(WEAK_RERANKER_WEIGHT).to_numpy() if category_weights else WEAK_RERANKER_WEIGHT
    return (1 - weights) * reranker_ranks + weights * category_ranks(weak_reranker, categories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--items_path", required=True)
    parser.add_argument("--matches_path", required=True)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.set_float32_matmul_precision("high")
    started = time.perf_counter()
    matches = pd.read_parquet(args.matches_path, columns=["id1", "id2"])
    required_ids = np.unique(matches[["id1", "id2"]].to_numpy().reshape(-1))
    items = relevant_items(args.items_path, required_ids)
    texts = item_texts(items)
    categories = matches["id1"].map(items.set_index("id")["category"])
    routed_categories = VERIFIED_CATEGORIES | AMBIGUOUS_CATEGORIES
    reranker = route_by_category(
        matches,
        categories,
        routed_categories,
        lambda rows, row_categories: reranker_predictions(rows, texts, MODEL_PATH, PRIMARY_MAX_LENGTH, PRIMARY_BATCH_SIZE, row_categories, PRIMARY_REVERSE_FRACTION, PRIMARY_REVERSE_FRACTIONS),
        lambda rows, row_categories: route_by_category(
            rows,
            row_categories,
            AMBIGUOUS_CATEGORIES,
            lambda verified_rows, verified_categories: reranker_predictions(verified_rows, texts, ADAPTED_MODEL_PATH, PRIMARY_MAX_LENGTH, PRIMARY_BATCH_SIZE, verified_categories, PRIMARY_REVERSE_FRACTION, PRIMARY_REVERSE_FRACTIONS),
            lambda ambiguous_rows, ambiguous_categories: reranker_predictions(ambiguous_rows, texts, AMBIGUOUS_MODEL_PATH, PRIMARY_MAX_LENGTH, PRIMARY_BATCH_SIZE, ambiguous_categories, PRIMARY_REVERSE_FRACTION, PRIMARY_REVERSE_FRACTIONS),
        ),
    )
    weak_reranker = reranker_predictions(matches, texts, WEAK_MODEL_PATH, SECONDARY_MAX_LENGTH, SECONDARY_BATCH_SIZE, categories, SECONDARY_REVERSE_FRACTION) if WEAK_MODEL_PATH.exists() else None
    output = matches.copy()
    output["predict"] = np.clip(blend(reranker, weak_reranker, categories, WEAK_RERANKER_WEIGHTS), 0, 1)
    output.to_csv(args.output_path, index=False)
    print(json.dumps({"rows": len(output), "items": len(items), "seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
