import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


REQUIRED_COLUMNS = ["text1", "text2", "target", "weight", "category"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_loss(scores, targets, minimum_gap=0.2):
    target_gap = targets[:, None] - targets[None, :]
    mask = target_gap >= minimum_gap
    if not mask.any():
        return scores.sum() * 0.0
    score_gap = scores[:, None] - scores[None, :]
    return F.softplus(-score_gap[mask]).mean()


def freeze_encoder_layers(model, count):
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    for layer in model.roberta.encoder.layer[:count]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


class PairDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame) * 2

    def __getitem__(self, index):
        row = self.frame.iloc[index % len(self.frame)]
        swap = index >= len(self.frame)
        return {
            "text1": row["text2"] if swap else row["text1"],
            "text2": row["text1"] if swap else row["text2"],
            "target": float(row["target"]),
            "weight": float(row["weight"]),
        }


class CategoryBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        categories = dataset.frame["category"].astype(str).tolist()
        self.groups = {}
        for index, category in enumerate(categories):
            self.groups.setdefault(category, []).extend([index, index + len(categories)])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            batches.extend(
                shuffled[offset:offset + self.batch_size]
                for offset in range(0, len(shuffled), self.batch_size)
            )
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


class PairCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        tokens = self.tokenizer(
            [row["text1"] for row in rows],
            [row["text2"] for row in rows],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens["targets"] = torch.tensor([row["target"] for row in rows], dtype=torch.float32)
        tokens["weights"] = torch.tensor([row["weight"] for row in rows], dtype=torch.float32)
        return tokens


def validate_frame(frame):
    if frame.columns.tolist() != REQUIRED_COLUMNS:
        raise ValueError("training columns differ from contract")
    if frame.empty:
        raise ValueError("training frame is empty")
    if not frame["target"].between(0, 1).all():
        raise ValueError("targets outside [0, 1]")
    if not frame["weight"].gt(0).all():
        raise ValueError("non-positive weights")
    if frame[["text1", "text2", "category"]].isna().any().any():
        raise ValueError("missing text or category")


def train(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(args.pairs)
    validate_frame(frame)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    freeze_encoder_layers(model, args.freeze_layers)
    model.to("cuda").train()
    dataset = PairDataset(frame)
    sampler = CategoryBatchSampler(dataset, args.batch_size, args.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=PairCollator(tokenizer, args.max_length))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * args.warmup_ratio)),
        num_training_steps=total_updates,
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    update = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            targets = batch.pop("targets").to("cuda", non_blocking=True)
            weights = batch.pop("weights").to("cuda", non_blocking=True)
            inputs = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.float().reshape(-1)
                pointwise = (
                    F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights
                ).sum() / weights.sum()
                ranking = rank_loss(logits, targets, args.minimum_rank_gap)
                loss = (pointwise + args.rank_weight * ranking) / args.gradient_accumulation
            loss.backward()
            boundary = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
            if not boundary:
                continue
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update == 1 or update % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "update": update,
                            "total_updates": total_updates,
                            "loss": float(loss.detach()) * args.gradient_accumulation,
                            "pointwise": float(pointwise.detach()),
                            "ranking": float(ranking.detach()),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    output.mkdir(parents=True)
    model.to(torch.bfloat16)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    manifest = {
        "pairs": str(Path(args.pairs).resolve()),
        "pairs_sha256": sha256(args.pairs),
        "source_model": str(Path(args.model).resolve()),
        "source_weights_sha256": sha256(Path(args.model) / "model.safetensors"),
        "rows": len(frame),
        "category_rows": frame["category"].value_counts().sort_index().to_dict(),
        "target_mean": float(frame["target"].mean()),
        "weight_sum": float(frame["weight"].sum()),
        "parameters": vars(args),
        "updates": update,
        "seconds": time.perf_counter() - started,
    }
    (output / "training-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", default="models/adapted_reranker")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--minimum-rank-gap", type=float, default=0.2)
    parser.add_argument("--freeze-layers", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026083006)
    parser.add_argument("--log-every", type=int, default=100)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
