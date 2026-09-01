import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WEAK_CATEGORIES = (
    "Обувь",
    "Одежда",
    "Галантерея и аксессуары",
    "Ювелирные изделия",
    "Мебель",
    "Спорт и отдых",
    "Канцелярские товары",
)
AMBIGUOUS_CATEGORIES = (
    "Галантерея и аксессуары",
    "Одежда",
    "Ювелирные изделия",
)
ROWS_PER_CATEGORY = 100_000
SEED = 2026083002
MAX_CHARS = 3600


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quoted(value):
    return str(value).replace("'", "''")


def excluded_sql(paths):
    if not paths:
        return "SELECT NULL::BIGINT AS left_id, NULL::BIGINT AS right_id WHERE FALSE"
    values = ", ".join(f"'{quoted(Path(path).resolve())}'" for path in paths)
    return f"""
        SELECT least(id1, id2) AS left_id, greatest(id1, id2) AS right_id
        FROM read_parquet([{values}])
    """


def selection_sql(items, matches_llm, excluded):
    categories = ", ".join(f"'{quoted(value)}'" for value in WEAK_CATEGORIES)
    return f"""
        COPY (
            WITH excluded AS ({excluded_sql(excluded)}),
            joined AS (
                SELECT
                    m.id1,
                    m.id2,
                    m.target,
                    i1.category,
                    CAST(i1.name AS VARCHAR) AS left_name,
                    CAST(i1.attributes AS VARCHAR) AS left_attributes,
                    CAST(i2.name AS VARCHAR) AS right_name,
                    CAST(i2.attributes AS VARCHAR) AS right_attributes,
                    CASE
                        WHEN lower(trim(CAST(i1.name AS VARCHAR))) = lower(trim(CAST(i2.name AS VARCHAR)))
                            AND CAST(i1.attributes AS VARCHAR) <> CAST(i2.attributes AS VARCHAR)
                            THEN 'exact_name'
                        WHEN m.target > 0.0 AND m.target < 0.7777777778 THEN 'ambiguous'
                        WHEN m.target >= 0.7777777778
                            AND lower(trim(CAST(i1.name AS VARCHAR))) <> lower(trim(CAST(i2.name AS VARCHAR)))
                            THEN 'high_vote'
                        WHEN m.target = 0.0
                            AND substr(lower(trim(CAST(i1.name AS VARCHAR))), 1, 14)
                                = substr(lower(trim(CAST(i2.name AS VARCHAR))), 1, 14)
                            THEN 'zero_prefix'
                        ELSE 'other'
                    END AS stratum,
                    hash(m.id1, m.id2, {SEED}) AS h
                FROM read_parquet('{quoted(Path(matches_llm).resolve())}') m
                JOIN read_parquet('{quoted(Path(items).resolve())}') i1 ON i1.id = m.id1
                JOIN read_parquet('{quoted(Path(items).resolve())}') i2 ON i2.id = m.id2
                WHERE i1.category IN ({categories})
                    AND NOT EXISTS (
                        SELECT 1 FROM excluded e
                        WHERE e.left_id = least(m.id1, m.id2)
                            AND e.right_id = greatest(m.id1, m.id2)
                    )
            ),
            ranked AS (
                SELECT *, row_number() OVER (PARTITION BY category, stratum ORDER BY h) AS stratum_rank
                FROM joined
            ),
            fixed AS (
                SELECT * FROM ranked
                WHERE (stratum = 'ambiguous' AND stratum_rank <= 25000)
                    OR (stratum = 'exact_name' AND stratum_rank <= 22000)
                    OR (stratum = 'high_vote' AND stratum_rank <= 20000)
                    OR (stratum = 'zero_prefix' AND stratum_rank <= 18000)
            ),
            remaining AS (
                SELECT category, {ROWS_PER_CATEGORY} - count(*) AS quota
                FROM fixed GROUP BY category
            ),
            selected AS (
                SELECT * FROM fixed
                UNION ALL
                SELECT ranked.* FROM ranked
                JOIN remaining USING (category)
                WHERE ranked.stratum = 'other' AND ranked.stratum_rank <= remaining.quota
            )
            SELECT id1, id2, target, category, left_name, left_attributes,
                right_name, right_attributes, stratum, h
            FROM selected
            ORDER BY category, stratum, h
        ) TO '{{output}}' (FORMAT PARQUET)
    """


def select_pairs(items, matches_llm, excluded, output):
    import duckdb

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sql = selection_sql(items, matches_llm, excluded).format(output=quoted(output))
    duckdb.sql(sql)
    frame = pd.read_parquet(output, columns=["category", "stratum"])
    counts = frame["category"].value_counts().to_dict()
    if counts != {category: ROWS_PER_CATEGORY for category in WEAK_CATEGORIES}:
        raise ValueError(f"category balance differs from contract: {counts}")
    return {
        "rows": len(frame),
        "category_rows": counts,
        "stratum_rows": frame.groupby(["category", "stratum"]).size().to_dict(),
        "output_sha256": sha256(output),
    }


def materialize_pairs(source, output):
    from structured_features import serialize_item

    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(source)
    result = pd.DataFrame(
        {
            "text1": [
                serialize_item(name, attributes, category, MAX_CHARS)
                for name, attributes, category in frame[["left_name", "left_attributes", "category"]].itertuples(index=False, name=None)
            ],
            "text2": [
                serialize_item(name, attributes, category, MAX_CHARS)
                for name, attributes, category in frame[["right_name", "right_attributes", "category"]].itertuples(index=False, name=None)
            ],
            "target": frame["target"].astype(float),
            "weight": 1.0,
            "category": frame["category"].astype(str),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return {"rows": len(result), "source_sha256": sha256(source), "output_sha256": sha256(output)}


def score_direction(model, tokenizer, left, right, batch_size, max_length):
    import torch

    scores = []
    with torch.inference_mode():
        for offset in range(0, len(left), batch_size):
            tokens = tokenizer(
                left[offset:offset + batch_size],
                right[offset:offset + batch_size],
                padding=True,
                truncation="longest_first",
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to("cuda", non_blocking=True) for key, value in tokens.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**tokens).logits.float().reshape(-1)
            scores.extend(logits.cpu().tolist())
    return scores


def score_pairs(pairs, model_path, output, batch_size, max_length):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(pairs)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to("cuda").eval()
    left = frame["text1"].tolist()
    right = frame["text2"].tolist()
    frame["reference_logit"] = score_direction(model, tokenizer, left, right, batch_size, max_length)
    frame["reference_logit_swap"] = score_direction(model, tokenizer, right, left, batch_size, max_length)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {"rows": len(frame), "pairs_sha256": sha256(pairs), "output_sha256": sha256(output)}


def mix_targets(teacher, forward_logits, reverse_logits, teacher_weight):
    if not 0 <= teacher_weight <= 1:
        raise ValueError("teacher weight outside [0, 1]")
    teacher = np.asarray(teacher, dtype=np.float64)
    forward = 1 / (1 + np.exp(-np.asarray(forward_logits, dtype=np.float64)))
    reverse = 1 / (1 + np.exp(-np.asarray(reverse_logits, dtype=np.float64)))
    preserved = (forward + reverse) / 2
    return teacher_weight * teacher + (1 - teacher_weight) * preserved


def build_mixture(logits, output, teacher_weight, categories):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(logits)
    frame["target"] = mix_targets(
        frame["target"],
        frame["reference_logit"],
        frame["reference_logit_swap"],
        teacher_weight,
    )
    if categories:
        frame = frame.loc[frame["category"].isin(categories)]
    frame = frame[["text1", "text2", "target", "weight", "category"]].reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    return {
        "rows": len(frame),
        "categories": frame["category"].value_counts().sort_index().to_dict(),
        "target_mean": float(frame["target"].mean()),
        "source_sha256": sha256(logits),
        "output_sha256": sha256(output),
    }


def write_receipt(output, values):
    path = Path(output).with_suffix(".manifest.json")
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--items", required=True)
    select.add_argument("--matches-llm", required=True)
    select.add_argument("--exclude", action="append", default=[])
    select.add_argument("--output", required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--source", required=True)
    materialize.add_argument("--output", required=True)
    score = commands.add_parser("score")
    score.add_argument("--pairs", required=True)
    score.add_argument("--model", default="models/reranker")
    score.add_argument("--output", required=True)
    score.add_argument("--batch-size", type=int, default=384)
    score.add_argument("--max-length", type=int, default=384)
    mix = commands.add_parser("mix")
    mix.add_argument("--logits", required=True)
    mix.add_argument("--output", required=True)
    mix.add_argument("--teacher-weight", type=float, default=0.5)
    mix.add_argument("--categories", nargs="*", default=AMBIGUOUS_CATEGORIES)
    args = parser.parse_args()
    if args.command == "select":
        receipt = select_pairs(args.items, args.matches_llm, args.exclude, args.output)
    elif args.command == "materialize":
        receipt = materialize_pairs(args.source, args.output)
    elif args.command == "score":
        receipt = score_pairs(args.pairs, args.model, args.output, args.batch_size, args.max_length)
    else:
        receipt = build_mixture(args.logits, args.output, args.teacher_weight, args.categories)
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
