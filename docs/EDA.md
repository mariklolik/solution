# Что показал EDA

Анализ был разделён по пяти исходным таблицам и их связям: `items.md`, `items_human.md`, `matches.md`, `matches_llm.md`, `cross_file_findings.md`. Ниже сохранены выводы, которые повлияли на финальный код; полный contest data в репозиторий не включён.

## `items.md`

- В полном каталоге 13 397 761 карточка.
- Найдено 1 806 552 строки с полностью совпадающим содержимым, объединённые в 614 575 групп. Самая большая группа содержит 6 002 карточки с общим названием вроде «палатка 2-местная».
- Совпадение текста не является надёжным признаком identity: общие названия скрывают размер, цвет, комплектность, OEM и совместимость.
- Есть 11 пустых названий; у большей части таких строк остаются информативные attributes. Blanket filtering теряет полезные карточки.
- В длинном хвосте attributes часто находятся model, OEM, fitment и числовые характеристики. Поэтому сериализация объединяет category, name и attributes до truncation.

## `items_human.md`

- Human-часть содержит 711 304 уникальных endpoint-карточки.
- ID human и LLM cohorts не пересекаются, но 2 999 exact-content групп связывают их по содержимому. Обычный случайный split по строкам переоценивает перенос; для диагностики использовались source-closed компоненты и signatures.
- Карточки одной пары часто происходят из разных marketplace schemas. Почти полное отсутствие одинаковых attribute keys делает сравнение по позиции поля ненадёжным; flat serialization устойчивее к смене schema.

## `matches.md`

- 365 654 ручные пары, 93 890 positives, общий positive rate 25.68%, 20 категорий.
- Pair graph почти целиком состоит из одиночных рёбер: 97.72% вершин имеют degree 1, triangles отсутствуют. Graph propagation и классический listwise grouping получают слишком мало структуры.
- Category prior меняется от 7.26% до 56.20%. Global threshold или один calibration mapping противоречат macro-метрике.
- Даже при одинаковом normalized title positive rate меняется по доменам: около 6.01% в аксессуарах, 7.19% в обуви, 12.84% в одежде и 95.69% в музыкальных инструментах.
- В 32 452 положительных human-парах есть хотя бы один numeric conflict. Числа полезны, но не могут быть жёстким veto без понимания поля.

## `matches_llm.md`

- 11 187 780 soft-labeled пар. Target лежит на сетке `k/9`: 7 184 663 строки имеют нулевой vote, 1 223 955 имеют один vote, 2 779 162 имеют промежуточные значения.
- 1 020 483 majority-positive пары содержат numeric conflicts. Ошибки не случайны: они концентрируются в personalization, optics, auto fitment и product-versus-accessory случаях.
- Teacher prevalence по категориям заметно отличается от human prevalence. Raw vote нельзя считать калиброванной вероятностью.

## `cross_file_findings.md`

Совместный вывод оказался важнее любой одной статистики. Большой teacher даёт покрытие, frozen RankNet сохраняет уже выученную границу, а category routing защищает домены от negative transfer. Поэтому target specialist равен `0.5 * official LLM vote + 0.5 * mean(sigmoid(base_forward), sigmoid(base_reverse))`; corpus сбалансирован по 100 000 строк на категорию; expert допускается только для категорий, где обе закрытые панели дали положительный результат. Именно EDA отверг единый threshold, общий seven-category continuation, graph propagation и перенос human OOF как универсального proxy.
