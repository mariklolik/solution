# Ambiguous Route2

### Category-aware self-distillation для быстрого product matching

**Официальный результат:** `0.5259256928`, статус `Success`, прирост `+0.0009972103` относительно direct RankNet. В контейнере организатора решение обработало 20 000 пар за 43.3 секунды. Во время inference сеть не используется.

## 1. Общая идея

В двадцати категориях одна и та же похожесть означает разные вещи. Совпадающее название музыкального инструмента часто указывает на один товар. Для одежды, обуви и аксессуаров за тем же названием могут скрываться другой размер, материал, совместимость или variant. Поэтому Ambiguous Route2 использует **selective plasticity**: каждая категория получает только ту степень adaptation, которая для неё оправдана. Category router направляет пятнадцать стабильных категорий в исходный direct RankNet, `Мебель` и `Обувь` в Route2 checkpoint, а `Галантерею и аксессуары`, `Одежду` и `Ювелирные изделия` в отдельный ambiguous specialist. Specialist обучен на мягкой цели `0.5 × official LLM vote + 0.5 × frozen RankNet probability`; вероятность RankNet усредняется для прямого и обратного порядка пары. Нижние 20 из 24 encoder layers заморожены, поэтому новое знание меняет верхние слои и classification head, сохраняя базовое представление товара. В результате получается компактный mixture-of-experts с прозрачным routing и заранее ограниченной областью воздействия каждого checkpoint.

```text
Category + Name + Attributes
              |
              v
       category router
     /         |          \
direct      Route2      ambiguous KD50
15 cats   Furniture,    Accessories,
          Shoes         Clothing, Jewelry
     \         |          /
      within-category rank
              |
     72.5% primary rank
   + 27.5% mMiniLM rank
              |
           predict
```

## 2. Данные и обучение

EDA показал, какие сигналы модель должна сохранять. В каталоге из 13 397 761 карточки есть 1 806 552 exact-content duplicate rows, но крупнейшая одинаковая группа содержит 6 002 общих карточки вроде «палатка 2-местная». Exact text здесь создаёт сильный, но небезопасный признак. Надёжность одинакового normalized title меняется ещё резче: 6.01% positives в аксессуарах, 7.19% в обуви, 12.84% в одежде и 95.69% в музыкальных инструментах. Human graph почти не содержит структуры для propagation: 97.72% вершин имеют degree 1, triangles отсутствуют. В 11 187 780 LLM-парах votes распределены по сетке `k/9`; среди majority-positive пар 1 020 483 содержат numeric conflicts. Ошибки сосредоточены в размерах, personalization, optics, auto fitment и различии между товаром и аксессуаром. Из этих наблюдений следуют четыре решения: сериализовать `Category`, `Name` и длинные `Attributes`; учить specialist отдельно по неоднозначным категориям; не трактовать LLM vote как calibrated probability; не использовать global threshold и жёсткие numeric veto. Training corpus содержит по 100 000 пар на каждую из трёх ambiguous-категорий. Soft BCE учит вероятность совпадения, pairwise loss сохраняет порядок внутри category-homogeneous batch. Точный pipeline подготовки данных и обучения находится в [`training/`](training/), а цифры из пяти EDA-отчётов собраны в [`docs/EDA.md`](docs/EDA.md).

## 3. Как работает inference

`run.py` читает пары и потоково проходит по `items.parquet`, оставляя только участвующие карточки. Это позволяет работать с полным каталогом без загрузки 13.4 млн строк в память. Карточка превращается в одну последовательность `Category: ... Name: ... Attributes: ...`; пары сортируются по суммарной длине, чтобы уменьшить padding. Затем router последовательно запускает нужные primary checkpoints. Для верхней половины score distribution внутри каждой категории считается и обратный порядок пары, после чего две вероятности усредняются. Этот дополнительный проход тратится на область, которая сильнее всего влияет на верх ranking. Absolute logits разных доменов напрямую не сравниваются: каждый primary score переводится во within-category percentile rank. Финальный ответ равен `0.725 × routed BGE rank + 0.275 × mMiniLM rank`. Компактный mMiniLM работает с `max_length=256` и batch `1024`, primary models с `max_length=384` и batch `384`. Checkpoints загружаются по очереди и освобождаются после своего маршрута. Такой forward path сохраняет сильный cross-encoder, добавляет дешёвую rank-level стабилизацию и укладывается в жёсткий runtime limit.

## 4. Научная основа и результат

Метод собран из нескольких проверяемых идей. В [jina-reranker-v3.5: An Efficient Listwise Reranker with Hybrid Attention and Self-Distillation](https://arxiv.org/html/2607.18152v1) для нас важны failure-mode-first curation, self-distillation между моделями одинакового размера и сохранение ranking при adaptation. В решение вошли эти принципы, но не Jina weights, hybrid attention или LBNL: используемые BGE и mMiniLM checkpoints имеют Apache-2.0 model cards. [Learning without Forgetting](https://arxiv.org/abs/1606.09282) мотивировала response preservation: frozen RankNet участвует в target и удерживает уже выученную функцию. [Born-Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html) обосновала same-capacity student, поэтому ambiguous expert остаётся в той же BGE architecture. [Distilling Virtual Examples for Long-Tailed Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/He_Distilling_Virtual_Examples_for_Long-Tailed_Recognition_ICCV_2021_paper.html) подсказала контролировать soft-label distribution; отсюда одинаковый объём данных по категориям. Финальная конфигурация прошла две независимые source-closed панели с положительной дельтой во всех трёх ambiguous-категориях, затем получила официальный `Success` и score `0.5259256928`. Код напрямую отражает метод: `mix_targets`, balanced sampling, frozen-depth continuation, category-homogeneous batches, pairwise rank loss и category routing. Ablations и границы доказательств приведены в [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Структура репозитория

| Файл | Ответственность |
|---|---|
| [`run.py`](run.py) | streaming I/O, primary routing, reverse scoring, rank blend |
| [`structured_features.py`](structured_features.py) | сериализация карточек |
| [`targeted_route.py`](targeted_route.py) | category masks и восстановление порядка пар |
| [`training/prepare_distillation.py`](training/prepare_distillation.py) | выбор пар, frozen-teacher scoring, KD50 target |
| [`training/train_specialist.py`](training/train_specialist.py) | обучение ambiguous specialist |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | полный forward path и runtime decisions |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | data preparation, training, inference, packaging |
| [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) | лицензии, lineage, соответствие правилам |

## Запуск

Веса извлекаются из [официально оценённого archive](https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/c1a51c4b/submission-ambiguous-route2-v1.zip):

```bash
curl -L https://storage.yandexcloud.net/ds-ods/files/submissions/0d0183cf-8864-4331-88be-f78da0c68dd2/c1a51c4b/submission-ambiguous-route2-v1.zip -o /tmp/ambiguous-route2.zip
unzip /tmp/ambiguous-route2.zip 'models/*' -d .
python -u run.py --items_path /data/items.parquet --matches_path /data/matches.parquet --output_path submission.csv
```

```bash
python -m scripts.package_submission --output outputs/submission-ambiguous-route2.zip
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Contest data в репозиторий не входит. Репозиторий должен оставаться private до письменного разрешения организатора на публикацию согласно [правилам E-CUP 2026](https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/E-CUP-2026/E-CUP-2026-Rules.pdf).
