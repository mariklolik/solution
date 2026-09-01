# Архитектура Ambiguous Route2

## Контракт

На вход приходят `items.parquet` и пары `id1,id2`. На выходе CSV с теми же `id1,id2` в исходном порядке и конечным `predict` в диапазоне `[0,1]`. Inference не использует сеть, внешние сервисы и файлы вне архива.

```text
items + candidate pairs
        |
        v
Category / Name / Attributes serialization
        |
        v
category router
  |                 |                       |
15 categories       Furniture, Shoes        Accessories, Clothing, Jewelry
direct RankNet       Route2 checkpoint       ambiguous KD50 specialist
  |                 |                       |
  +-----------------+-----------------------+
                    |
           top-half reverse scoring
                    |
          within-category percentile rank
                    |
        72.5% routed BGE + 27.5% mMiniLM rank
                    |
                 predict
```

## Почему три primary-маршрута

`models/reranker` является неизменяемым direct RankNet anchor. `models/adapted_reranker` применяется только к категориям `Мебель` и `Обувь`, где перенос был подтверждён официальным покатегорийным результатом. `models/ambiguous_reranker` применяется только к `Галантерея и аксессуары`, `Одежда` и `Ювелирные изделия`, где отдельный specialist прошёл две source-closed панели. Остальные пятнадцать категорий не зависят от specialist checkpoint.

Router находится в `targeted_route.py`. Он вычисляет маски, запускает scorer на нужных строках и восстанавливает исходный порядок. Вложенный вызов отделяет два verified-домена от трёх ambiguous-доменов.

## Scoring

Каждая карточка сериализуется как `Category`, `Name`, `Attributes`. Primary reranker использует `max_length=384`, `batch_size=384` и bf16. Для половины строк с наибольшим forward-score внутри категории дополнительно считается обратный порядок пары; две вероятности усредняются. Это направляет второй forward pass туда, где он сильнее влияет на верхнюю часть ranking.

После routing absolute logits не смешиваются между категориями. Каждый scorer переводится во within-category percentile rank. Финальный score равен `0.725 * routed_primary_rank + 0.275 * weak_rank`. Компактный mMiniLM работает с `max_length=256`, `batch_size=1024` и стабилизирует общий порядок при небольшой стоимости.

## Память и время

`run.py` сначала читает пары, затем потоково сканирует `items.parquet` батчами по 262144 строки и оставляет только участвующие ID. Пары сортируются по суммарной длине текстов перед tokenization, что уменьшает padding. Модели маршрутов загружаются последовательно и освобождаются до загрузки следующей. Сеть во время inference не используется.

## Где находится логика

| Файл | Ответственность |
|---|---|
| `run.py` | I/O, selective item scan, inference, routing, rank blend |
| `targeted_route.py` | category masks и восстановление порядка |
| `structured_features.py` | canonical serialization; дополнительные deterministic pair features сохранены как исследовательский слой |
| `training/prepare_distillation.py` | target-blind selection, base scoring, KD target mixture |
| `training/train_specialist.py` | frozen-depth, category-homogeneous BCE + pairwise ranking continuation |
| `scripts/package_submission.py` | allowlist-сборка ZIP без research-файлов и caches |
| `scripts/verify_submission.py` | проверка SHA-256, состава и CRC |
