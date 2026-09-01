# Эксперименты и границы доказательств

## Официальные результаты

| Вариант | Public macro PR-AUC | Статус |
|---|---:|---|
| Direct RankNet control | 0.5249284824 | официальный anchor |
| Route2 | 0.5258686762 | `Мебель`, `Обувь` routed |
| Ambiguous direct-init | 0.5254291625 | отвергнут |
| Ambiguous Route2 | **0.5259256928** | официальный `Success` |

Ambiguous Route2 улучшил direct RankNet на `+0.0009972103` и Route2 на `+0.0000570165`. Последняя дельта мала. Сильная часть результата заключается в том, что routing превратил неоднородный transfer в контролируемое изменение пяти категорий и сохранил рабочий anchor на остальных пятнадцати.

## Почему появился Route2

Первый six-category route дал общий официальный score 0.5248218565 и был отвергнут. Его покатегорийные дельты раскрыли причину:

| Категория | Official delta |
|---|---:|
| Мебель | +0.0099599323 |
| Обувь | +0.0089588160 |
| Галантерея и аксессуары | -0.0086673305 |
| Канцелярские товары | -0.0064672648 |
| Ювелирные изделия | -0.0030811997 |
| Одежда | -0.0027764399 |

Route2 оставил adapted checkpoint только в первых двух категориях. Это дало официальный `+0.0009401938` относительно direct RankNet.

## Specialist grid

На общем 700k corpus были зафиксированы до просмотра метрик teacher-only, KD25, KD50, KD75, freeze 8/12/18/20, learning rate `5e-7/1e-6/2e-6`, rank weight `0/0.5/1.0`, direct и Route2 initialization. Общий checkpoint для семи категорий не прошёл transfer gate. Отдельные `Furniture`, `Shoes` и `Stationery + Sport` specialists не улучшили уже выбранные маршруты достаточно устойчиво.

Единственным согласованным блоком стали `Галантерея и аксессуары`, `Одежда`, `Ювелирные изделия`. Для финального KD50 Route2-init source-closed deltas были положительны во всех трёх категориях:

| Frozen panel | Аксессуары | Одежда | Ювелирка | Macro |
|---|---:|---:|---:|---:|
| Manual extreme AP | +0.005952 | +0.007597 | +0.006774 | +0.006774 |
| Teacher extreme AP | +0.005408 | +0.008046 | +0.005289 | +0.006248 |

Это локальные диагностические панели, не официальный score. Их задача состояла в защите от regression и выборе кандидата, а не в прогнозе leaderboard.

## Выбранное обучение

- Init: Route2 adapted checkpoint.
- Данные: 300 000 строк, ровно 100 000 на ambiguous-категорию.
- Soft target: 50% official LLM vote, 50% frozen direct RankNet probability, усреднённая по направлениям.
- Один epoch, lower 20/24 encoder layers frozen, LR `1e-6`, batch 256, pairwise weight 0.5, max length 384.
- 2 346 optimizer updates, 831.8 секунды на исходном training run.

## Runtime evidence

Exact-container smoke обработал 20 000 пар и 39 722 карточки за 43.297 секунды. CSV schema, ID order, finite range, archive CRC и file hashes прошли. Официальный terminal status `Success` подтверждает, что uploaded artifact был принят и завершил evaluation; smoke не выдаётся за отдельный full-private benchmark.
