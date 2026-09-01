# Соответствие правилам E-CUP 2026

Источник: [официальные правила E-CUP 2026](https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/E-CUP-2026/E-CUP-2026-Rules.pdf). Репозиторий должен оставаться private до получения письменного разрешения организатора.

| Требование | Реализация и evidence |
|---|---|
| 2.8: законные права на third-party software, возможность коммерческого применения | Primary основан на `BAAI/bge-reranker-v2-m3`, weak scorer на `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; обе model cards указывают Apache-2.0. Jina v3.5 weights не используются, поскольку их license non-commercial. |
| 2.11: только open-license LLM для самостоятельной переразметки; proprietary LLM запрещены | Самостоятельная LLM-переразметка не выполнялась. Specialist обучен на официальном `matches_llm` vote и prediction включённого frozen RankNet. Proprietary API, внешние labels и hidden targets отсутствуют. |
| 2.11: код обучения и разметки воспроизводим | `training/prepare_distillation.py`, `training/train_specialist.py`, manifests, hashes и команды находятся в репозитории. |
| 2.12: самостоятельная разработка, запрет публикации без разрешения, запрет публикации contest data | GitHub repository private. Contest Parquet/CSV не коммитятся и исключены `.gitignore`. Код и receipts созданы для этой работы; public solution code не копировался. |
| Runtime contract | `metadata.json` использует официальный image `odsai/ecup26-matching-baseline:1.0` и `entry_point: python -u run.py`. Inference не делает network calls. |

## Third-party assets

| Каталог | Upstream | License |
|---|---|---|
| `models/reranker`, `models/adapted_reranker`, `models/ambiguous_reranker` | [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) и собственные fine-tunes | Apache-2.0 |
| `models/weak_reranker` | [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | Apache-2.0 |

## Данные

- Training source: только organizer-provided `items.parquet` и `matches_llm.parquet`.
- Human `matches.parquet` и две source-closed панели исключены из specialist train.
- Evaluation panels не входят в репозиторий и не используются как labels.
- Repo не содержит карточки, пары, private IDs, leaderboard targets или bearer tokens.
