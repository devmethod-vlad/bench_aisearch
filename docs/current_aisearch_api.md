# Бенчмарки под текущий aisearch

Добавлены новые скрипты рядом со старыми, чтобы не ломать существующие сценарии:

- `metrics/metrics_bench_current.py` — замер времени и API-метрик.
- `quality/quality_bench_current.py` — оценка качества выдачи.
- `bench_common/current_api.py` — общий клиент для `POST /hybrid-search/search` и `GET /hybrid-search/info/{task_id}`.

## Что изменилось в контракте aisearch

Текущий `aisearch` принимает runtime-параметры в теле поискового запроса:

- `search_use_cache`
- `metrics_enable`
- `show_intermediate_results`
- `presearch`
- `filters`

Поэтому новые скрипты отправляют эти поля явно. Для честного замера времени по умолчанию cache выключен, метрики включены.

## Основные переменные окружения

```env
API_BASE_URL=http://localhost:5155
API_SEARCH_PATH=/hybrid-search/search
API_STATUS_PATH=/hybrid-search/info/{task_id}
API_TOP_K=10
TOP_K=10
API_SEARCH_USE_CACHE=false
API_METRICS_ENABLE=true
API_SHOW_INTERMEDIATE_RESULTS=true
RESULTS_PATH=info.results
METRICS_PATH=info.metrics
INTERMEDIATE_RESULTS_PATH=info.intermediate_results
RESULT_ID_FIELD=ext_id
MARGIN_SCORE_FIELD=
```

Если `MARGIN_SCORE_FIELD` пустой, скрипт сам пробует поля `final_score`, `score_final`, `score_ce`, `score_fusion`, `score_dense`, `score_lex`.

## Quality benchmark

Минимальный набор колонок Excel:

```env
TEST_DATA_PATH=test_data.xlsx
COL_QUERY=test_query
COL_TARGET_ID=target_id
COL_SOURCE=source
COL_QUERY_SOURCE=test_query_source
COL_ANSWER=test_answer
METRICS_KS=1,3,5,10
OUTPUT_BASENAME=bench_results_current
```

Запуск:

```bash
python -m quality.quality_bench_current
```

Результат: JSON и XLSX с `recall@k`, `MRR@k`, `nDCG@k`, рангами, task_id и временем ответа.

## Metrics benchmark

Можно прогнать один запрос несколько раз:

```env
TEST_QUERY=тестовый запрос
TEST_QUERY_RETRY_COUNT=20
```

Или взять запросы из Excel/Parquet:

```env
SOURCE_FILE=test_data.xlsx
SOURCE_FILE_QUERY_FIELD=test_query
```

Запуск:

```bash
python -m metrics.metrics_bench_current
```

Результат: JSON с полными ответами API и агрегатами по `response_time`, `total_time`, `embedding_time`, `vector_search_time`, `lexical_search_time`, `cross_encoder_time`.

## Фильтры и presearch

Передаются как JSON-строки:

```env
API_PRESEARCH_JSON={"field":"question"}
API_FILTERS_JSON={"array_filters":{"role":["Врач"]},"exact_filters":{"source":"kb"}}
```

Формат полностью повторяет `SearchRequest` текущего `aisearch`.
