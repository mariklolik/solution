import json
import math
import re

import numpy as np
import pandas as pd


TOKEN_PATTERN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
MODEL_PATTERN = re.compile(r"(?=[a-zа-яё0-9._/-]{3,})(?=[a-zа-яё0-9._/-]*[a-zа-яё])(?=[a-zа-яё0-9._/-]*\d)[a-zа-яё0-9]+(?:[._/-][a-zа-яё0-9]+)*", re.IGNORECASE)
BRAND_KEYS = ("бренд", "марка", "brand", "manufacturer", "производитель")
IDENTIFIER_KEYS = ("артикул", "модель", "model", "part", "партномер", "код товара", "sku", "mpn")


def serialize_item(name, attributes, category, max_chars):
    try:
        parsed = json.loads(attributes)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {}
    values = []
    for key, value in parsed.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        values.append(f"{key}: {value}")
    return f"Category: {category} Name: {name} Attributes: {'; '.join(values)}"[:max_chars]


def normalize(value):
    return " ".join(TOKEN_PATTERN.findall(str(value).casefold().replace("ё", "е")))


def limited_set(values, limit=256):
    result = set()
    for value in values:
        result.add(value)
        if len(result) == limit:
            break
    return result


def tokens(value, limit=256):
    return limited_set(TOKEN_PATTERN.findall(str(value).casefold().replace("ё", "е")), limit)


def numbers(value):
    result = set()
    for raw in NUMBER_PATTERN.findall(str(value).casefold()):
        normalized = raw.replace(",", ".")
        try:
            number = float(normalized)
            normalized = f"{number:.8g}"
        except ValueError:
            pass
        result.add(normalized)
        if len(result) == 128:
            break
    return result


def models(value):
    return limited_set((part.casefold().replace("ё", "е") for part in MODEL_PATTERN.findall(str(value))), 128)


def trigrams(value):
    compact = normalize(value).replace(" ", "_")[:512]
    return {compact[index:index + 3] for index in range(max(0, len(compact) - 2))}


def similarities(left, right):
    overlap = len(left & right)
    union = len(left | right)
    smaller = min(len(left), len(right))
    larger = max(len(left), len(right))
    return (
        overlap / union if union else 1.0,
        overlap / smaller if smaller else float(not larger),
        overlap / larger if larger else 1.0,
        math.log1p(overlap),
    )


def parsed_attributes(value):
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    normalized = {normalize(key): normalize(item) for key, item in parsed.items() if item is not None}
    text = " ".join(f"{key} {item}" for key, item in normalized.items())
    return normalized, text


def selected_attribute_text(attributes, fragments):
    return " ".join(value for key, value in attributes.items() if any(fragment in key for fragment in fragments))


def pair_features(name1, attributes1, category1, name2, attributes2, category2):
    clean_name1 = normalize(name1)
    clean_name2 = normalize(name2)
    parsed1, attribute_text1 = parsed_attributes(attributes1)
    parsed2, attribute_text2 = parsed_attributes(attributes2)
    name_tokens1, name_tokens2 = tokens(clean_name1), tokens(clean_name2)
    name_numbers1, name_numbers2 = numbers(name1), numbers(name2)
    name_models1, name_models2 = models(name1), models(name2)
    attribute_keys1, attribute_keys2 = set(parsed1), set(parsed2)
    attribute_tokens1, attribute_tokens2 = tokens(attribute_text1), tokens(attribute_text2)
    combined1 = f"{clean_name1} {attribute_text1}"
    combined2 = f"{clean_name2} {attribute_text2}"
    combined_tokens1, combined_tokens2 = tokens(combined1), tokens(combined2)
    combined_numbers1, combined_numbers2 = numbers(combined1), numbers(combined2)
    combined_models1, combined_models2 = models(combined1), models(combined2)
    brand1 = tokens(selected_attribute_text(parsed1, BRAND_KEYS))
    brand2 = tokens(selected_attribute_text(parsed2, BRAND_KEYS))
    identifier1 = models(selected_attribute_text(parsed1, IDENTIFIER_KEYS)) | name_models1
    identifier2 = models(selected_attribute_text(parsed2, IDENTIFIER_KEYS)) | name_models2
    common_keys = attribute_keys1 & attribute_keys2
    exact_common_values = sum(parsed1[key] == parsed2[key] for key in common_keys)
    nonempty_names = bool(clean_name1 and clean_name2)
    features = [
        float(category1 == category2),
        float(nonempty_names and clean_name1 == clean_name2),
        float(nonempty_names and (clean_name1 in clean_name2 or clean_name2 in clean_name1)),
        min(len(clean_name1), len(clean_name2)) / max(1, max(len(clean_name1), len(clean_name2))),
        abs(len(clean_name1) - len(clean_name2)) / max(1, max(len(clean_name1), len(clean_name2))),
    ]
    for left, right in [
        (name_tokens1, name_tokens2),
        (trigrams(name1), trigrams(name2)),
        (name_numbers1, name_numbers2),
        (name_models1, name_models2),
    ]:
        features.extend(similarities(left, right))
    features.extend([
        float(bool(name_numbers1 and name_numbers2) and not bool(name_numbers1 & name_numbers2)),
        float(bool(name_models1 and name_models2) and not bool(name_models1 & name_models2)),
        float(normalize(attributes1) == normalize(attributes2)),
        min(len(str(attributes1)), len(str(attributes2))) / max(1, max(len(str(attributes1)), len(str(attributes2)))),
    ])
    for left, right in [
        (attribute_keys1, attribute_keys2),
        (attribute_tokens1, attribute_tokens2),
        (combined_tokens1, combined_tokens2),
        (combined_numbers1, combined_numbers2),
        (combined_models1, combined_models2),
        (brand1, brand2),
        (identifier1, identifier2),
    ]:
        features.extend(similarities(left, right))
    features.extend([
        float(bool(brand1 and brand2) and not bool(brand1 & brand2)),
        float(bool(identifier1 and identifier2) and not bool(identifier1 & identifier2)),
        len(common_keys) / max(1, len(attribute_keys1 | attribute_keys2)),
        exact_common_values / max(1, len(common_keys)),
        math.log1p(len(common_keys)),
        math.log1p(exact_common_values),
    ])
    return features


def feature_names(categories):
    names = ["same_category", "exact_name", "contained_name", "name_length_ratio", "name_length_gap"]
    for prefix in ["name_tokens", "name_trigrams", "name_numbers", "name_models"]:
        names.extend(f"{prefix}_{suffix}" for suffix in ["jaccard", "containment", "coverage", "overlap"])
    names.extend(["name_number_conflict", "name_model_conflict", "exact_attributes", "attribute_length_ratio"])
    for prefix in ["attribute_keys", "attribute_tokens", "combined_tokens", "combined_numbers", "combined_models", "brand", "identifier"]:
        names.extend(f"{prefix}_{suffix}" for suffix in ["jaccard", "containment", "coverage", "overlap"])
    names.extend(["brand_conflict", "identifier_conflict", "common_key_ratio", "exact_common_value_ratio", "common_key_count", "exact_common_value_count"])
    names.extend(f"category={category}" for category in categories)
    return names


def make_features(matches, items, categories=None, progress=25000):
    categories = categories or sorted(items["category"].astype(str).unique())
    category_index = {category: index for index, category in enumerate(categories)}
    left = items.rename(columns={column: f"{column}1" for column in ["id", "name", "attributes", "category"]})
    right = items.rename(columns={column: f"{column}2" for column in ["id", "name", "attributes", "category"]})
    pairs = matches.merge(left, on="id1", how="left", validate="many_to_one").merge(right, on="id2", how="left", validate="many_to_one")
    if pairs[["name1", "name2"]].isna().any().any():
        raise ValueError("Missing item rows")
    base_names = feature_names([])
    matrix = np.zeros((len(pairs), len(base_names) + len(categories)), dtype=np.float32)
    columns = ["name1", "attributes1", "category1", "name2", "attributes2", "category2"]
    for index, row in enumerate(pairs[columns].itertuples(index=False, name=None)):
        values = pair_features(*row)
        matrix[index, :len(values)] = values
        matrix[index, len(values) + category_index[str(row[2])]] = 1.0
        if progress and (index + 1) % progress == 0:
            print(json.dumps({"features": index + 1, "rows": len(pairs)}), flush=True)
    return matrix, feature_names(categories), categories
