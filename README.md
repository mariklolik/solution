# Ambiguous Route2

### Category-aware self-distillation для product matching

> Официальный результат E-CUP 2026: `0.5259256928`, статус `Success`. В контейнере организатора 20 000 пар обработаны за 43.3 секунды без доступа к сети.

## Решение за четыре абзаца

Product matching в большом каталоге нельзя свести к одной шкале похожести. Одинаковое название музыкального инструмента почти всегда указывает на один товар, а в одежде за тем же названием могут скрываться другой размер, материал или модель. Ambiguous Route2 учитывает это прямо в архитектуре. Category router оставляет 15 стабильных категорий на исходном RankNet, отправляет `Мебель` и `Обувь` в Route2 checkpoint, а для `Галантереи и аксессуаров`, `Одежды` и `Ювелирных изделий` включает отдельный specialist. Каждый checkpoint меняет только свой участок каталога, поэтому domain adaptation остаётся локальной и управляемой.

Такой routing следует из структуры данных. Среди 13 397 761 карточки нашлось 1 806 552 полных текстовых дубля, однако крупнейшая группа из 6 002 карточек содержала общее название уровня «палатка 2-местная». Даже точное совпадение normalized title даёт радикально разную вероятность match: 6.01% в аксессуарах, 7.19% в обуви, 12.84% в одежде и 95.69% в музыкальных инструментах. Human graph тоже не помогает построить универсальное правило: 97.72% вершин имеют degree 1, triangles отсутствуют. Поэтому модель читает `Category`, `Name` и `Attributes` до общего лимита 3 600 символов, а решения принимает внутри категории.

Ambiguous specialist обучен на 300 000 парах, поровну из трёх сложных категорий. Его soft target, KD50, смешивает два разных источника знания: `0.5 × official LLM vote + 0.5 × frozen RankNet probability`. RankNet probability усредняется для обоих порядков пары, чтобы target не зависел от того, какая карточка стоит слева. LLM даёт широкое покрытие каталога, frozen RankNet сохраняет уже выученную границу, а сбалансированная выборка не позволяет крупной категории задавить две остальные. Нижние 20 из 24 encoder layers заморожены; обучаются верхние слои и head. Objective объединяет soft BCE с pairwise RankNet loss в category-homogeneous batches.

Inference повторяет ту же логику без лишней инфраструктуры. `run.py` потоково сканирует `items.parquet`, сохраняет только нужные карточки, сортирует пары по длине и запускает checkpoints последовательно, освобождая GPU-память между маршрутами. Для верхней половины score distribution внутри категории дополнительно считается обратный порядок пары. Затем logits переводятся во within-category percentile rank, после чего routed BGE rank смешивается с быстрым mMiniLM rank в пропорции `0.725 / 0.275`. Весь метод сводится к одному прозрачному forward path, который помещается в конкурсный runtime.

```text
Category + Name + Attributes
              |
              v
       category router
      /        |         \
 direct     Route2     KD50 specialist
 15 cats    2 cats        3 cats
      \        |         /
       within-category rank
              |
    72.5% routed BGE rank
      + 27.5% mMiniLM rank
              |
           predict
```

## Что именно показал EDA

Пять срезов EDA свелись к трём решениям, которые видны в коде.

Первое: exact text нельзя использовать как shortcut. Числа, размеры, OEM, fitment и совместимость часто лежат в длинном хвосте `Attributes`, поэтому [`structured_features.py`](structured_features.py) собирает карточку из всех трёх полей до truncation.

Второе: teacher target нельзя читать как calibrated probability. В 11 187 780 LLM-парах значения лежат на сетке `k/9`, а 1 020 483 majority-positive пары содержат numeric conflicts. Эти конфликты особенно часты в personalization, optics, auto fitment и случаях «товар против аксессуара». Поэтому решение не использует один global threshold и не вводит жёсткий numeric veto.

Третье: категория определяет смысл похожести сильнее, чем общий prior. Отсюда отдельный specialist, равные квоты по 100 000 пар и category routing вместо одного checkpoint на весь каталог. Полная сводка измерений находится в [`docs/EDA.md`](docs/EDA.md).

## На каких работах стоит метод

- [jina-reranker-v3.5](https://arxiv.org/html/2607.18152v1) формулирует три близкие идеи: failure-mode-first curation, multi-domain mixture и self-distillation между моделями одного размера. В Ambiguous Route2 они проявляются как выбор сложных категорий по EDA, отдельный specialist и distillation без compression. Hybrid attention, LBNL и Jina weights здесь не используются.
- [Learning without Forgetting](https://arxiv.org/abs/1606.09282) показывает, как сохранить ответы исходной модели при adaptation. Поэтому половина KD50 target приходит от замороженного RankNet, а большая часть encoder остаётся frozen.
- [Born-Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html) рассматривает distillation без уменьшения student. Specialist сохраняет ту же BGE architecture: цель обучения состоит в переносе soft supervision, а не в сжатии.
- [Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) задаёт pairwise основу RankNet. В training loop она сохраняет относительный порядок пар внутри категории, пока soft BCE отвечает за сам score.
- [Distilling Virtual Examples for Long-Tailed Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/He_Distilling_Virtual_Examples_for_Long-Tailed_Recognition_ICCV_2021_paper.html) связывает distillation с распределением soft labels в long-tail данных. Практический вывод для этого решения прост: распределение supervision нужно контролировать, поэтому каждая ambiguous-категория получает одинаковую квоту.

## Где смотреть код

| Файл | Что в нём |
|---|---|
| [`run.py`](run.py) | streaming I/O, model routing, reverse scoring и rank blend |
| [`targeted_route.py`](targeted_route.py) | category masks и восстановление исходного порядка пар |
| [`structured_features.py`](structured_features.py) | сериализация карточек и исследовательские pair features |
| [`training/prepare_distillation.py`](training/prepare_distillation.py) | выбор пар, frozen-teacher scoring и сборка KD50 target |
| [`training/train_specialist.py`](training/train_specialist.py) | frozen-depth continuation, soft BCE и pairwise loss |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | полный forward path и решения по runtime |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | параметры обучения и подтверждённые результаты |

## Запуск

Веса лежат в [официально оценённом archive](https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/c1a51c4b/submission-ambiguous-route2-v1.zip):

```bash
curl -L https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/c1a51c4b/submission-ambiguous-route2-v1.zip -o /tmp/ambiguous-route2.zip
unzip /tmp/ambiguous-route2.zip 'models/*' -d .
python -u run.py \
  --items_path /data/items.parquet \
  --matches_path /data/matches.parquet \
  --output_path submission.csv
```

Сборка submission и тесты:

```bash
python -m scripts.package_submission --output outputs/submission-ambiguous-route2.zip
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Contest data в репозиторий не входит. Подготовка данных, обучение и упаковка описаны в [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
