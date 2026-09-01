# Воспроизводимость

## 1. Получить веса

Weights извлекаются из официально оценённого archive. После clone:

```bash
curl -L https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/c1a51c4b/submission-ambiguous-route2-v1.zip -o /tmp/ambiguous-route2.zip
unzip /tmp/ambiguous-route2.zip 'models/*' -d .
python -m scripts.verify_submission
```

Команда сверяет 26 packaged-файлов с receipt `artifacts/scored-artifact.json`. Runtime и веса из archive совпадают с официально scored submission по SHA-256. Исходный ZIP имеет SHA-256 `513cd6c889e73f8ac067b8554da37d8724275bbb0dd3ed1385b29f0e265071d0`.

## 2. Подготовить финальный training corpus

Все входы ниже являются файлами организатора или закрытыми evaluation exclusions. Репозиторий не содержит contest data.

```bash
python -m training.prepare_distillation select \
  --items /data/items.parquet \
  --matches-llm /data/matches_llm.parquet \
  --exclude /data/matches.parquet \
  --exclude /eval/manual-source-closed-20k/matches.parquet \
  --exclude /eval/teacher-source-closed-20k/matches.parquet \
  --output outputs/private-700k.parquet

python -m training.prepare_distillation materialize \
  --source outputs/private-700k.parquet \
  --output outputs/teacher-pairs.parquet

python -m training.prepare_distillation score \
  --pairs outputs/teacher-pairs.parquet \
  --model models/reranker \
  --output outputs/base-logits.parquet

python -m training.prepare_distillation mix \
  --logits outputs/base-logits.parquet \
  --teacher-weight 0.5 \
  --output outputs/ambiguous-kd50-pairs.parquet
```

Selection использует seed `2026083002`, семь категорий по 100 000 строк и пять predeclared strata. Из 700k после `mix` остаются три ambiguous-категории, 300k строк. Ожидаемые lineage hashes записаны в `artifacts/training-receipt.json`.

## 3. Обучить ambiguous specialist

```bash
python -m training.train_specialist \
  --pairs outputs/ambiguous-kd50-pairs.parquet \
  --model models/adapted_reranker \
  --output outputs/ambiguous_reranker
```

Defaults совпадают с выбранным run: one epoch, batch 256, freeze 20, LR `1e-6`, rank weight 0.5, seed `2026083006`. Trainer удваивает corpus перестановкой сторон, формирует category-homogeneous batches, оптимизирует weighted soft BCE плюс pairwise ranking loss и сохраняет bf16 Safetensors вместе с manifest.

Полная bitwise идентичность повторного CUDA training run не обещается из-за GPU kernels. Dataset hashes, source checkpoint, parameters, code и scored output weights зафиксированы. Точный direct RankNet, Route2 anchors и solution-specific specialist берутся из scored archive; specialist continuation воспроизводится командой выше.

## 4. Запустить inference

В образе организатора:

```bash
python -u run.py \
  --items_path /data/items.parquet \
  --matches_path /data/matches.parquet \
  --output_path outputs/submission.csv
```

## 5. Собрать ZIP

```bash
python -m scripts.package_submission --output outputs/submission-ambiguous-route2.zip
python -m scripts.verify_submission --zip outputs/submission-ambiguous-route2.zip
```

Packager использует явный allowlist. `docs`, `tests`, `training`, Git metadata, caches и contest data в ZIP не попадают.
