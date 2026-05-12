# bench_aisearch

Два независимых бенчмарка для `devmethod-vlad/aisearch`:
- `metrics` — замер latency и API-метрик.
- `quality` — оценка качества выдачи (recall@k, MRR@k, nDCG@k, rank, not_found).

Общий клиент текущего API: `bench_common/current_api.py`.

## Подготовка env

У каждого бенчмарка свой env-файл:
- metrics: `.env_metrics` (пример: `.env_metrics.example`)
- quality: `.env_quality` (пример: `.env_quality.example`)

```bash
cp .env_metrics.example .env_metrics
cp .env_quality.example .env_quality
```

## Запуск

Metrics benchmark:

```bash
python -m metrics.metrics_bench
```

Quality benchmark:

```bash
python -m quality.quality_bench
```

## Переопределение dotenv-файла

```bash
DOTENV_PATH=.env_metrics.local python -m metrics.metrics_bench
DOTENV_PATH=.env_quality.local python -m quality.quality_bench
```

## Отличия входных данных

- Metrics benchmark: либо `SOURCE_FILE` + `SOURCE_FILE_QUERY_FIELD`, либо `TEST_QUERY` + `TEST_QUERY_RETRY_COUNT`.
- Quality benchmark: обязателен `TEST_DATA_PATH` и колонки `COL_QUERY` / `COL_TARGET_ID` / `COL_SOURCE`.

Детальный контракт API и полный список переменных: `docs/current_aisearch_api.md`.
