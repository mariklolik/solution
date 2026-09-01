# Claim-to-source ledger

## Research

| Claim | Primary source | Что реально использовано |
|---|---|---|
| Same-size self-distillation может служить для domain adaptation, а failure-mode-first curation важнее общего объёма | [jina-reranker-v3.5](https://arxiv.org/html/2607.18152v1) | Принцип staged adaptation, same-size teacher/student и отдельный corpus для наблюдаемого failure mode. Hybrid attention, listwise LBNL и Jina weights не используются. |
| Сохранение response старой модели уменьшает forgetting на новом домене | [Learning without Forgetting](https://arxiv.org/abs/1606.09282) | Frozen direct RankNet probability входит в KD50 target. |
| Student той же ёмкости может получить пользу от soft teacher outputs | [Born-Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html) | Specialist использует тот же BGE reranker backbone, distillation не является compression. |
| Soft targets полезны для long-tail режимов, но их distribution нужно контролировать | [Distilling Virtual Examples for Long-Tailed Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/He_Distilling_Virtual_Examples_for_Long-Tailed_Recognition_ICCV_2021_paper.html) | Ровно 100k строк на категорию и soft target mixture; effective-number weighting не добавлялся поверх уже balanced corpus. |

## Models and rules

| Claim | Source |
|---|---|
| BGE reranker is multilingual, 0.6B, Apache-2.0 | [BAAI model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| mMiniLM cross-encoder covers 15 languages and is Apache-2.0 | [Sentence Transformers model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) |
| Contest restrictions and pitch criteria | [E-CUP 2026 rules](https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/E-CUP-2026/E-CUP-2026-Rules.pdf) |

## Local evidence

| Claim | Receipt |
|---|---|
| Exact file hashes, official score, submission ID, smoke | `artifacts/scored-artifact.json` |
| Dataset and checkpoint lineage, training parameters | `artifacts/training-receipt.json` |
| Route6, KD alpha and init ablations | `artifacts/ablation-receipt.json` |
| Exact specialist training run | `models/ambiguous_reranker/training-manifest.json` |

Official metrics, source-closed metrics and runtime smoke are reported separately. Ни одна local panel delta не называется leaderboard score.
